#!/usr/bin/env python3
"""Validate and time one-call ordinary-dot BF16-W8 SwiGLU AOT."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from triton_bf16_w8_backend import (
    TritonBF16W8MLPBackend,
    pack_w8_blocks,
)


def median_us(function, warmup: int, iterations: int, batches: int) -> float:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(batches):
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - begin) / iterations / 1000.0)
    return statistics.median(samples)


def paired_median_us(
    first, second, warmup: int, iterations: int, batches: int
) -> tuple[float, float]:
    for _ in range(warmup):
        first()
        second()
    first_samples = []
    second_samples = []

    def measure(function) -> float:
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        return (time.perf_counter_ns() - begin) / iterations / 1000.0

    for batch in range(batches):
        if batch % 2 == 0:
            first_samples.append(measure(first))
            second_samples.append(measure(second))
        else:
            second_samples.append(measure(second))
            first_samples.append(measure(first))
    return statistics.median(first_samples), statistics.median(second_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--quant-dir", required=True)
    parser.add_argument("--gemv-dir", required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument(
        "--candidate-dir",
        help="optional directory containing a second quant/GEMV/activation set",
    )
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(816)
    torch.set_num_threads(1)
    x = torch.randn(args.k, dtype=torch.bfloat16) * 0.2
    gate_weight = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    up_weight = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    packed = pack_w8_blocks(
        torch.cat((gate_weight, up_weight), dim=1), args.block_n
    )
    gate_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    up_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    scale = torch.cat((gate_scale, up_scale)).contiguous()
    output = torch.empty(args.n, dtype=torch.bfloat16)
    candidate_output = torch.empty_like(output)
    backend = TritonBF16W8MLPBackend(
        args.library,
        args.quant_dir,
        args.gemv_dir,
        args.activation_dir,
        args.k,
        args.n,
        args.block_n,
    )

    def run() -> None:
        backend(x, packed, scale, output)

    run()
    xf = x.float()
    absmax = xf.abs().max().clamp(min=1.0e-8)
    x_scale = absmax / 127.0
    x_q = (xf * (127.0 / absmax)).clamp(-128, 127).round().to(torch.int8)
    gate = (
        (x_q.to(torch.int32) @ gate_weight.to(torch.int32)).float()
        * x_scale
        * gate_scale
    ).to(torch.bfloat16)
    up = (
        (x_q.to(torch.int32) @ up_weight.to(torch.int32)).float()
        * x_scale
        * up_scale
    ).to(torch.bfloat16)
    expected = (F.silu(gate) * up).to(torch.bfloat16)
    assert torch.equal(output, expected)

    candidate_backend = None
    if args.candidate_dir:
        candidate_backend = TritonBF16W8MLPBackend(
            args.library,
            args.candidate_dir,
            args.candidate_dir,
            args.candidate_dir,
            args.k,
            args.n,
            args.block_n,
        )

        def run_candidate() -> None:
            candidate_backend(x, packed, scale, candidate_output)

        run_candidate()
        assert torch.equal(candidate_output, expected)
        baseline_us, candidate_us = paired_median_us(
            run, run_candidate, args.warmup, args.iters, args.batches
        )
        print(
            "PASS BF16-W8 ordinary AOT SwiGLU paired A/B\n"
            f"baseline_us={baseline_us:.3f}\n"
            f"candidate_us={candidate_us:.3f}\n"
            f"speedup={baseline_us / candidate_us:.4f}x\n"
            "bit_exact=True"
        )
        candidate_backend.close()
        backend.close()
        return

    latency = median_us(run, args.warmup, args.iters, args.batches)
    print(
        "PASS BF16-W8 ordinary AOT SwiGLU\n"
        f"one_ctypes_call_us={latency:.3f}\n"
        f"bit_exact={torch.equal(output, expected)}"
    )
    backend.close()


if __name__ == "__main__":
    main()
