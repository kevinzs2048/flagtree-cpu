#!/usr/bin/env python3
"""Microbenchmark MiniCPM BF16 lm_head versus production Q4-G32/Q8 Triton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from flag_gems.runtime.backend._arm.q4.linear import (
    prepare_w8_weight,
    prepare_weight_asym,
)


def median_call_us(function, iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    return (time.perf_counter_ns() - start) / iterations / 1000.0


def accuracy(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    delta = actual_f - reference_f
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(delta * delta))),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.clamp(torch.linalg.vector_norm(reference_f), min=1.0e-30)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                actual_f.reshape(1, -1), reference_f.reshape(1, -1)
            )[0]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=("q4-g32", "w8-channel"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/cix/MiniCPM5-2.6B-W4A8-GPTQ-G128"),
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(
            "artifacts/vllm-libtriton-jit-q4/build/"
            "libtriton_jit_q4_op.so"
        ),
    )
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.m < 1 or args.iterations < 1 or args.batches < 3:
        raise ValueError("M/iterations must be positive and batches>=3")

    from safetensors import safe_open

    torch.set_num_threads(args.threads)
    torch.ops.load_library(str(args.library.resolve()))
    with safe_open(
        args.model.resolve() / "model.safetensors",
        framework="pt",
        device="cpu",
    ) as checkpoint:
        weight = checkpoint.get_tensor("lm_head.weight").contiguous()
    n, k = weight.shape
    torch.manual_seed(20260809)
    x = torch.randn((args.m, k), dtype=torch.bfloat16)

    pack_start = time.perf_counter()
    if args.format == "q4-g32":
        rhs = prepare_weight_asym(weight)

        def generated() -> torch.Tensor:
            return torch.ops.triton_jit_cpu.q4_linear_g32_asym(x, rhs, n, k)

        packed_bytes = rhs.numel() * rhs.element_size()
    else:
        decode_rhs, prefill_rhs, scale = prepare_w8_weight(weight)

        def generated() -> torch.Tensor:
            return torch.ops.triton_jit_cpu.w8_linear(
                x, decode_rhs, prefill_rhs, scale, n, k
            )

        packed_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (decode_rhs, prefill_rhs, scale)
        )
    pack_s = time.perf_counter() - pack_start

    def aten_bf16() -> torch.Tensor:
        return torch.nn.functional.linear(x, weight)

    reference = aten_bf16()
    actual = generated()
    metrics = accuracy(actual, reference)
    for _ in range(5):
        generated()
        aten_bf16()

    functions = {"triton_codegen": generated, "aten_bf16": aten_bf16}
    samples = {name: [] for name in functions}
    names = list(functions)
    for batch in range(args.batches):
        order = names[batch % 2 :] + names[: batch % 2]
        for name in order:
            samples[name].append(
                median_call_us(functions[name], args.iterations)
            )
    medians = {
        name: float(statistics.median(values))
        for name, values in samples.items()
    }
    result = {
        "format": args.format,
        "model": str(args.model.resolve()),
        "m": args.m,
        "n": n,
        "k": k,
        "threads": torch.get_num_threads(),
        "pack_s": pack_s,
        "source_weight_bytes": weight.numel() * weight.element_size(),
        "packed_weight_bytes": packed_bytes,
        "accuracy_vs_aten_bf16": metrics,
        "median_us": medians,
        "speedup_vs_aten_bf16": medians["aten_bf16"] / medians["triton_codegen"],
        "samples_us": samples,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
