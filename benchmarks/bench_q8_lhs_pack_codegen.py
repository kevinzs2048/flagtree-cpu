#!/usr/bin/env python3
"""A/B the masked and full-row Q8 KAI activation pack specializations."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = Path(
    os.getenv(
        "TRITON_CPU_VENV_SITE",
        "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
    )
)
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402
import triton  # noqa: E402

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    _pack_lhs_w8_i8mm_kai_kernel,
)


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
        if batch % 2:
            candidate_samples.append(measure(candidate))
            baseline_samples.append(measure(baseline))
        else:
            baseline_samples.append(measure(baseline))
            candidate_samples.append(measure(candidate))
    return statistics.median(baseline_samples), statistics.median(
        candidate_samples
    )


def audit(compiled) -> dict[str, int]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()

    def instruction_count(opcode: str) -> int:
        return sum(
            line.lstrip().split(None, 1)[0] == opcode
            for line in assembly.splitlines()
            if line.lstrip()
        )

    return {
        "asm_lines": len(assembly.splitlines()),
        "umax": instruction_count("umax"),
        "fmaxnm": instruction_count("fmaxnm"),
        "fminnm": instruction_count("fminnm"),
        "folded_spills": assembly.count("folded spill"),
        "folded_reloads": assembly.count("folded reload"),
        "external_calls": sum(
            " call " in line and "llvm." not in line and " asm " not in line
            for line in llir.splitlines()
        ),
    }


def check_all_finite_uniform_contract() -> None:
    """Prove clamp removal for every finite BF16 magnitude and sign."""
    bits = torch.arange(0x7F80, dtype=torch.int32).to(torch.uint16)
    positive = bits.view(torch.bfloat16).float()
    for values in (positive, -positive):
        scale = values.abs().clamp_min(1.0e-8) / 127.0
        scaled = values / scale
        assert scaled.abs().max() < 128.0
        clamped = scaled.clamp(-128.0, 127.0).to(torch.int8)
        assert torch.equal(scaled.to(torch.int8), clamped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batches", type=int, default=11)
    args = parser.parse_args()
    if args.m <= 0 or args.m % 4 or args.k % 32:
        raise ValueError("requires M>0, M%4=0 and K%32=0")

    torch.set_num_threads(1)
    torch.manual_seed(5239)
    panel_stride = 4 * args.k + 16
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    masked = torch.empty((args.m // 4) * panel_stride, dtype=torch.int8)
    full = torch.empty_like(masked)
    grid = (args.m,)

    def run_masked() -> None:
        _pack_lhs_w8_i8mm_kai_kernel[grid](
            x,
            masked,
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
        )

    def run_full() -> None:
        _pack_lhs_w8_i8mm_kai_kernel[grid](
            x,
            full,
            args.m,
            x.stride(0),
            K=args.k,
            FULL_ROWS=True,
            num_warps=1,
            num_stages=1,
        )

    run_masked()
    run_full()
    if not torch.equal(masked, full):
        raise AssertionError("full-row specialization changed the packed blob")

    normal = x.clone()
    edge_bits = torch.tensor(
        [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F80,
            0xBF80,
            0x3F81,
            0xBF81,
            0x7F7F,
            0xFF7F,
        ],
        dtype=torch.uint16,
    )
    x_bits = x.view(torch.uint16)
    for begin in range(0, edge_bits.numel(), args.m):
        x_bits.zero_()
        for row, value in enumerate(edge_bits[begin : begin + args.m]):
            x_bits[row].fill_(value)
        run_masked()
        run_full()
        if not torch.equal(masked, full):
            raise AssertionError("finite BF16 edge pack differs")
    x.copy_(normal)
    check_all_finite_uniform_contract()
    masked_compiled = _pack_lhs_w8_i8mm_kai_kernel.warmup(
        x,
        masked,
        args.m,
        x.stride(0),
        K=args.k,
        num_warps=1,
        num_stages=1,
        grid=grid,
    )
    full_compiled = _pack_lhs_w8_i8mm_kai_kernel.warmup(
        x,
        full,
        args.m,
        x.stride(0),
        K=args.k,
        FULL_ROWS=True,
        num_warps=1,
        num_stages=1,
        grid=grid,
    )
    masked_us, full_us = paired_median_us(
        run_masked, run_full, args.warmup, args.iters, args.batches
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "triton_module": triton.__file__,
                "m": args.m,
                "k": args.k,
                "bit_exact_blob": True,
                "masked_us": masked_us,
                "full_rows_us": full_us,
                "full_over_masked": full_us / masked_us,
                "finite_bf16_edges_bit_exact": True,
                "all_finite_uniform_clamp_proof": True,
                "masked_codegen": audit(masked_compiled),
                "full_rows_codegen": audit(full_compiled),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
