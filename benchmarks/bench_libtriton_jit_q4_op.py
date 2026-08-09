#!/usr/bin/env python3
"""Paired microbenchmark of Q4 Python and libtriton_jit launch paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from flag_gems.runtime.backend._arm.q4.linear import (
    linear_w4a8,
    linear_w4a8_asym_kai,
    linear_w4a8_asym_g128_kai,
    pack_rhs_qsi4c32p,
    pack_rhs_qsi4c32p_asym,
    pack_rhs_qsi4c128p_asym,
    quantize_q4_0,
)


def median_us(function, iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    return (time.perf_counter_ns() - start) / iterations / 1000.0


def load_checkpoint_weight(
    checkpoint: Path, weight_key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    scale_key = weight_key.removesuffix(".weight") + ".weight_scale"
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if weight_key in keys:
                if scale_key not in keys:
                    raise KeyError(f"missing {scale_key} in {shard}")
                return (
                    handle.get_tensor(weight_key).contiguous(),
                    handle.get_tensor(scale_key).to(torch.bfloat16).contiguous(),
                )
    raise KeyError(f"{weight_key} not found below {checkpoint}")


def load_checkpoint_weights(
    checkpoint: Path, weight_keys: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one production projection or concatenate a fused vLLM group."""
    pairs = [load_checkpoint_weight(checkpoint, key) for key in weight_keys]
    k_values = {int(weight.shape[1]) for weight, _ in pairs}
    group_values = {int(scale.shape[1]) for _, scale in pairs}
    if len(k_values) != 1 or len(group_values) != 1:
        raise ValueError("joined checkpoint weights must share K and group count")
    return (
        torch.cat([weight for weight, _ in pairs], dim=0).contiguous(),
        torch.cat([scale for _, scale in pairs], dim=0).contiguous(),
    )


def accuracy(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float().reshape(-1)
    reference_f = reference.float().reshape(-1)
    delta = actual_f - reference_f
    relative_l2 = torch.linalg.vector_norm(delta) / torch.clamp(
        torch.linalg.vector_norm(reference_f), min=1.0e-30
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_f[None, :], reference_f[None, :]
    )[0]
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(delta * delta))),
        "relative_l2": float(relative_l2),
        "cosine_similarity": float(cosine),
        "bf16_mismatch_fraction": float((actual != reference).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2560)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, default=11)
    parser.add_argument("--g128", action="store_true")
    parser.add_argument("--asymmetric-g32", action="store_true")
    parser.add_argument("--decode-partitions", type=int)
    parser.add_argument("--decode-unroll", type=int, choices=(1, 2, 4))
    parser.add_argument("--compare-decode-unroll", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--weight-key",
        dest="weight_keys",
        action="append",
        help=(
            "checkpoint tensor to benchmark; repeat to concatenate a fused "
            "vLLM projection (default: layer-0 q_proj)"
        ),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.g128 and args.asymmetric_g32:
        raise ValueError("--g128 and --asymmetric-g32 are mutually exclusive")
    if args.decode_partitions is not None:
        if args.decode_partitions <= 0:
            raise ValueError("--decode-partitions must be positive")
        os.environ["FLAGGEMS_Q4_DECODE_PARTITIONS"] = str(
            args.decode_partitions
        )
    if args.decode_unroll is not None:
        os.environ["FLAGGEMS_Q4_DECODE_UNROLL"] = str(args.decode_unroll)
    torch.set_num_threads(args.threads)
    torch.ops.load_library(str(args.library.resolve()))
    torch.manual_seed(29)
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    if args.g128:
        if args.checkpoint is not None:
            weight_keys = args.weight_keys or [
                "model.layers.0.self_attn.q_proj.weight"
            ]
            quantized, scale = load_checkpoint_weights(
                args.checkpoint.resolve(), weight_keys
            )
            args.n, args.k = quantized.shape
            x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
            source = {
                "checkpoint": str(args.checkpoint.resolve()),
                "weight_key": weight_keys[0] if len(weight_keys) == 1 else None,
                "weight_keys": weight_keys,
            }
        else:
            quantized = torch.randint(-8, 8, (args.n, args.k), dtype=torch.int8)
            scale = (
                torch.rand(
                    (args.n, args.k // 128), dtype=torch.float32
                )
                * 0.08
                + 0.002
            ).to(torch.bfloat16)
            source = {
                "checkpoint": None,
                "weight_key": None,
                "weight_keys": None,
            }
        if args.k % 128:
            raise ValueError("--g128 requires K divisible by 128")
        if quantized.min() < -8 or quantized.max() > 7:
            raise ValueError("G128 checkpoint weight is not signed INT4")
        rhs = pack_rhs_qsi4c128p_asym(quantized, scale)
        unsigned = quantized.add(8)
        native_nibbles = (
            unsigned[:, 1::2] << 4 | unsigned[:, ::2]
        ).to(torch.uint8)
        native_rhs = torch.ops.aten._dyn_quant_pack_4bit_weight(
            native_nibbles, scale, None, 128, args.k, args.n
        )

        def native() -> torch.Tensor:
            return torch.ops.aten._dyn_quant_matmul_4bit(
                x.float(), native_rhs, 128, args.k, args.n
            ).to(torch.bfloat16)

        implementations = {
            "python_direct": lambda: linear_w4a8_asym_g128_kai(
                x, rhs, args.n, args.k
            ),
            "libtriton_jit_cpp_op": lambda: (
                torch.ops.triton_jit_cpu.q4_linear_g128(
                    x, rhs, args.n, args.k
                )
            ),
            "vllm_aten_kleidiai": native,
        }
        if args.compare_decode_unroll:
            if args.m != 1:
                raise ValueError("--compare-decode-unroll requires --m 1")
            implementations["libtriton_jit_cpp_op_u1"] = implementations[
                "libtriton_jit_cpp_op"
            ]
            implementations["libtriton_jit_cpp_op_u2"] = implementations[
                "libtriton_jit_cpp_op"
            ]
    else:
        if args.checkpoint is not None:
            if not args.asymmetric_g32:
                raise ValueError("a G32 checkpoint requires --asymmetric-g32")
            weight_keys = args.weight_keys or ["lm_head.weight"]
            quantized, scale = load_checkpoint_weights(
                args.checkpoint.resolve(), weight_keys
            )
            args.n, args.k = quantized.shape
            x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
            source = {
                "checkpoint": str(args.checkpoint.resolve()),
                "weight_key": weight_keys[0] if len(weight_keys) == 1 else None,
                "weight_keys": weight_keys,
            }
        else:
            source = {
                "checkpoint": None,
                "weight_key": None,
                "weight_keys": None,
            }
            weight = torch.randn((args.n, args.k), dtype=torch.bfloat16) * 0.15
            quantized, scale = quantize_q4_0(weight)

        if args.asymmetric_g32:
            rhs = pack_rhs_qsi4c32p_asym(
                quantized, scale.to(torch.bfloat16)
            )
            unsigned = quantized.add(8)
            native_nibbles = (
                unsigned[:, 1::2] << 4 | unsigned[:, ::2]
            ).to(torch.uint8)
            native_rhs = torch.ops.aten._dyn_quant_pack_4bit_weight(
                native_nibbles,
                scale.to(torch.bfloat16),
                None,
                32,
                args.k,
                args.n,
            )

            def native() -> torch.Tensor:
                return torch.ops.aten._dyn_quant_matmul_4bit(
                    x.float(), native_rhs, 32, args.k, args.n
                ).to(torch.bfloat16)

            implementations = {
                "python_direct": lambda: linear_w4a8_asym_kai(
                    x, rhs, args.n, args.k
                ),
                "libtriton_jit_cpp_op": lambda: (
                    torch.ops.triton_jit_cpu.q4_linear_g32_asym(
                        x, rhs, args.n, args.k
                    )
                ),
                "vllm_aten_kleidiai": native,
            }
        else:
            rhs = pack_rhs_qsi4c32p(quantized, scale)
            implementations = {
                "python_direct": lambda: linear_w4a8(x, rhs, args.n, args.k),
                "python_custom_op": lambda: torch.ops.flag_gems.arm_q4_linear(
                    x, rhs, args.n, args.k
                ),
                "libtriton_jit_cpp_op": lambda: (
                    torch.ops.triton_jit_cpu.q4_linear(
                        x, rhs, args.n, args.k
                    )
                ),
            }
    def select_unroll(name: str) -> None:
        if name.endswith("_u1"):
            os.environ["FLAGGEMS_Q4_DECODE_UNROLL"] = "1"
        elif name.endswith("_u2"):
            os.environ["FLAGGEMS_Q4_DECODE_UNROLL"] = "2"
        elif name == "libtriton_jit_cpp_op":
            if args.decode_unroll is None:
                os.environ.pop("FLAGGEMS_Q4_DECODE_UNROLL", None)
            else:
                os.environ["FLAGGEMS_Q4_DECODE_UNROLL"] = str(
                    args.decode_unroll
                )

    outputs = {}
    for name, function in implementations.items():
        select_unroll(name)
        outputs[name] = function()
    reference = outputs["python_direct"]
    for name, output in outputs.items():
        if name == "vllm_aten_kleidiai":
            continue
        if not torch.equal(reference, output):
            raise AssertionError(f"{name} is not BF16 bit-exact")
    for _ in range(20):
        for name, function in implementations.items():
            select_unroll(name)
            function()

    samples = {name: [] for name in implementations}
    names = list(implementations)
    for batch in range(args.batches):
        order = names[batch % len(names) :] + names[: batch % len(names)]
        for name in order:
            select_unroll(name)
            samples[name].append(
                median_us(implementations[name], args.iterations)
            )
    medians = {
        name: float(statistics.median(values))
        for name, values in samples.items()
    }
    result = {
        **source,
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "threads": torch.get_num_threads(),
        "decode_partitions": args.decode_partitions,
        "decode_unroll": args.decode_unroll,
        "format": (
            "g128"
            if args.g128
            else "g32_qai8dxp"
            if args.asymmetric_g32
            else "q4_0"
        ),
        "iterations": args.iterations,
        "batches": args.batches,
        "triton_paths_bf16_bit_exact": True,
        "aten_max_abs_error": (
            float((outputs["vllm_aten_kleidiai"].float() - reference.float()).abs().max())
            if "vllm_aten_kleidiai" in outputs
            else None
        ),
        "libtriton_jit_vs_vllm_kleidiai_accuracy": (
            accuracy(
                outputs["libtriton_jit_cpp_op"],
                outputs["vllm_aten_kleidiai"],
            )
            if "vllm_aten_kleidiai" in outputs
            else None
        ),
        "median_us": medians,
        "cpp_over_python_direct": (
            medians["libtriton_jit_cpp_op"] / medians["python_direct"]
        ),
        "cpp_over_python_custom_op": (
            medians["libtriton_jit_cpp_op"] / medians["python_custom_op"]
            if "python_custom_op" in medians
            else None
        ),
        "samples_us": samples,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
