#!/usr/bin/env python3
"""Paired W8 microbenchmark: Triton launch, libtriton_jit, and vLLM oneDNN.

All routes consume the same INT8 weight, FP32 per-channel scale, and BF16
activation.  Weight preparation is deliberately outside the timed region.
Use ``--checkpoint`` and ``--weight-key`` to benchmark a real model tensor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time


os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")
os.environ.setdefault("VLLM_PLUGINS", "fl")
os.environ.setdefault("FL_CPU_INT4", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    linear_w8_vllm_dynamic,
    pack_weights_i8mm_kai,
    pack_weights_sdot,
    pack_weights_sdot_blocked,
    select_w8_decode_tile_n,
)
from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_qsi8cxp
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
    CPUInt8ScaledMMLinearKernel,
)


class _NativeLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            scale.reshape(-1, 1).clone(), requires_grad=False
        )
        self.input_scale = None
        self.input_zero_point = None
        self.azp_adj = None
        self.logical_widths = [weight.shape[0]]
        self.bias = None


def _median_us(function, iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    return (time.perf_counter_ns() - start) / iterations / 1000.0


def _load_checkpoint_weight(
    checkpoint: Path, weight_key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    scale_key = weight_key.removesuffix(".weight") + ".weight_scale"
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if weight_key in keys:
                if scale_key not in keys:
                    raise KeyError(f"missing scale tensor {scale_key} in {shard}")
                return (
                    handle.get_tensor(weight_key).contiguous(),
                    handle.get_tensor(scale_key).reshape(-1).float().contiguous(),
                )
    raise KeyError(f"{weight_key} not found below {checkpoint}")


def _load_checkpoint_weights(
    checkpoint: Path, weight_keys: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one projection or concatenate the tensors fused by vLLM."""
    pairs = [_load_checkpoint_weight(checkpoint, key) for key in weight_keys]
    k_values = {int(weight.shape[1]) for weight, _ in pairs}
    if len(k_values) != 1:
        raise ValueError("joined checkpoint weights must share K")
    return (
        torch.cat([weight for weight, _ in pairs], dim=0).contiguous(),
        torch.cat([scale for _, scale in pairs], dim=0).contiguous(),
    )


def _accuracy(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float().reshape(-1)
    reference_f = reference.float().reshape(-1)
    delta = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f)
    relative_l2 = torch.linalg.vector_norm(delta) / torch.clamp(
        reference_norm, min=1.0e-30
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
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, default=11)
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
    if args.m < 1 or args.iterations < 1 or args.batches < 1:
        raise ValueError("M, iterations, and batches must be positive")

    torch.set_num_threads(args.threads)
    torch.manual_seed(43)
    if args.checkpoint is not None:
        weight_keys = args.weight_keys or [
            "model.layers.0.self_attn.q_proj.weight"
        ]
        weight, scale = _load_checkpoint_weights(
            args.checkpoint.resolve(), weight_keys
        )
        n, k = weight.shape
        source = {
            "checkpoint": str(args.checkpoint.resolve()),
            "weight_key": weight_keys[0] if len(weight_keys) == 1 else None,
            "weight_keys": weight_keys,
        }
    else:
        n, k = args.n, args.k
        weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
        scale = (
            torch.rand((n,), dtype=torch.float32) * 0.02 + 0.0005
        ).contiguous()
        source = {
            "checkpoint": None,
            "weight_key": None,
            "weight_keys": None,
        }
    if n % 64 or k % 32:
        raise ValueError(f"W8 test requires N%64=0 and K%32=0, got {n}/{k}")

    x = torch.randn((args.m, k), dtype=torch.bfloat16)
    torch.ops.load_library(str(args.library.resolve()))

    decode_block_n = select_w8_decode_tile_n(n, 64)
    decode_rhs = pack_weights_sdot_blocked(
        pack_weights_sdot(weight.T.contiguous()), decode_block_n
    ).contiguous()
    prefill_rhs = pack_weights_i8mm_kai(weight, scale).contiguous()
    guide_rhs = pack_rhs_qsi8cxp(weight, scale)

    native_layer = _NativeLayer(weight, scale)
    native_config = Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=False,
        is_channelwise=True,
        input_symmetric=True,
    )
    native_kernel = CPUInt8ScaledMMLinearKernel(
        native_config,
        [
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ],
    )
    native_kernel.process_weights_after_loading(native_layer)

    implementations = {
        "python_direct": lambda: linear_w8_vllm_dynamic(
            x, decode_rhs, prefill_rhs, scale, n, k
        ),
        "libtriton_jit_cpp_op": lambda: torch.ops.triton_jit_cpu.w8_linear(
            x, decode_rhs, prefill_rhs, scale, n, k
        ),
        "libtriton_jit_qai8dxp": lambda: (
            torch.ops.triton_jit_cpu.w8_linear_kai(x, guide_rhs, n, k)
        ),
        "vllm_onednn": lambda: native_kernel.apply_weights(native_layer, x),
    }
    outputs = {name: function() for name, function in implementations.items()}
    if not torch.equal(outputs["python_direct"], outputs["libtriton_jit_cpp_op"]):
        raise AssertionError("Python Triton and libtriton_jit outputs differ")

    for _ in range(20):
        for function in implementations.values():
            function()
    samples = {name: [] for name in implementations}
    names = list(implementations)
    for batch in range(args.batches):
        order = names[batch % len(names) :] + names[: batch % len(names)]
        for name in order:
            samples[name].append(
                _median_us(implementations[name], args.iterations)
            )
    medians = {
        name: float(statistics.median(values))
        for name, values in samples.items()
    }
    result = {
        **source,
        "m": args.m,
        "n": n,
        "k": k,
        "threads": torch.get_num_threads(),
        "iterations": args.iterations,
        "batches": args.batches,
        "python_and_cpp_bf16_bit_exact": True,
        "libtriton_jit_vs_vllm_onednn_accuracy": _accuracy(
            outputs["libtriton_jit_cpp_op"], outputs["vllm_onednn"]
        ),
        "qai8dxp_vs_vllm_onednn_accuracy": _accuracy(
            outputs["libtriton_jit_qai8dxp"], outputs["vllm_onednn"]
        ),
        "median_us": medians,
        "libtriton_jit_over_onednn": (
            medians["libtriton_jit_cpp_op"] / medians["vllm_onednn"]
        ),
        "libtriton_jit_over_python_direct": (
            medians["libtriton_jit_cpp_op"] / medians["python_direct"]
        ),
        "qai8dxp_over_onednn": (
            medians["libtriton_jit_qai8dxp"] / medians["vllm_onednn"]
        ),
        "qai8dxp_over_symmetric_codegen": (
            medians["libtriton_jit_qai8dxp"]
            / medians["libtriton_jit_cpp_op"]
        ),
        "samples_us": samples,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
