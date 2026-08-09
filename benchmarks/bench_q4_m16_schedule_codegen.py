#!/usr/bin/env python3
"""Paired M16-versus-two-M8 Q4 matrix schedule benchmark."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
sys.path[:0] = [str(ROOT / "third_party/FlagGems/src"), str(TRITON_PYTHON)]

import torch
import triton


def require_expected_triton() -> None:
    expected = (TRITON_PYTHON / "triton").resolve()
    actual = Path(triton.__file__).resolve()
    if expected not in actual.parents:
        raise RuntimeError(
            "wrong Triton source loaded: "
            f"expected under {expected}, got {actual}. "
            "Put TRITON_CPU_PYTHON at the front of PYTHONPATH before "
            "starting Python."
        )


require_expected_triton()

from flag_gems.runtime.backend._arm.q4.kernels import (
    _pack_lhs_qsi8d32p_panel4_scalar_kernel,
    _q4_prefill_i8mm_kai_kernel,
)
from flag_gems.runtime.backend._arm.q4.linear import prepare_weight


def paired_median_us(
    baseline, candidate, warmup: int, iterations: int, batches: int
) -> tuple[float, float]:
    for _ in range(warmup):
        baseline()
        candidate()

    def measure(function) -> float:
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        return (time.perf_counter_ns() - begin) / iterations / 1000.0

    baseline_samples = []
    candidate_samples = []
    for batch in range(batches):
        if batch % 2 == 0:
            baseline_samples.append(measure(baseline))
            candidate_samples.append(measure(candidate))
        else:
            candidate_samples.append(measure(candidate))
            baseline_samples.append(measure(baseline))
    return statistics.median(baseline_samples), statistics.median(
        candidate_samples
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=15)
    args = parser.parse_args()
    if args.m <= 0 or args.m % 16 or args.n % 4 or args.k % 32:
        raise ValueError("requires M%16=0, N%4=0 and K%32=0")

    torch.set_num_threads(1)
    torch.manual_seed(7223)
    m = args.m
    x = torch.randn((m, args.k), dtype=torch.bfloat16)
    weight = torch.randn((args.n, args.k), dtype=torch.bfloat16)
    rhs = prepare_weight(weight)
    groups = args.k // 32
    lhs = torch.empty((m // 4) * groups * 136, dtype=torch.uint8)
    m16_output = torch.empty((m, args.n), dtype=torch.bfloat16)
    m8_output = torch.empty_like(m16_output)

    _pack_lhs_qsi8d32p_panel4_scalar_kernel[(m // 4,)](
        x,
        lhs.view(torch.float16),
        lhs.view(torch.int8),
        m,
        x.stride(0),
        K=args.k,
        FULL_PANEL=True,
        num_warps=1,
        num_stages=1,
    )

    def run_m16() -> None:
        _q4_prefill_i8mm_kai_kernel[(m // 16, args.n // 4)](
            lhs.view(torch.int8),
            lhs.view(torch.float16),
            rhs.view(torch.uint8),
            rhs.view(torch.float16),
            m16_output,
            N=args.n,
            K=args.k,
            BLOCK_M=16,
            num_warps=1,
            num_stages=1,
        )

    def run_two_m8() -> None:
        _q4_prefill_i8mm_kai_kernel[(m // 8, args.n // 4)](
            lhs.view(torch.int8),
            lhs.view(torch.float16),
            rhs.view(torch.uint8),
            rhs.view(torch.float16),
            m8_output,
            N=args.n,
            K=args.k,
            BLOCK_M=8,
            num_warps=1,
            num_stages=1,
        )

    run_m16()
    run_two_m8()
    assert torch.equal(m8_output, m16_output)
    m16_us, m8_us = paired_median_us(
        run_m16, run_two_m8, args.warmup, args.iters, args.batches
    )
    print(
        f"PASS Q4 M16 schedule A/B M={m} N={args.n} K={args.k}\n"
        f"m16_us={m16_us:.3f}\n"
        f"two_m8_us={m8_us:.3f}\n"
        f"speedup={m16_us / m8_us:.4f}x\n"
        "bit_exact=True"
    )


if __name__ == "__main__":
    main()
