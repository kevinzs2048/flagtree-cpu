#!/usr/bin/env python3
"""Microbenchmark the exact production Q4 router, including LHS packing."""

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
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402
import triton  # noqa: E402


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

from flag_gems.runtime.backend._arm.q4.linear import (  # noqa: E402
    _make_vllm_linear,
    _use_m8_main_block,
    linear_w4a8,
    pack_rhs_qsi4c32p,
    quantize_q4_0,
    set_fused_decode_enabled,
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
    legacy, fused, warmup: int, iterations: int, batches: int
) -> dict[str, object]:
    for function in (legacy, fused):
        for _ in range(warmup):
            function()
    legacy_samples = []
    fused_samples = []
    ratios = []
    for batch in range(batches):
        order = (legacy, fused) if batch % 2 == 0 else (fused, legacy)
        elapsed = {}
        for function in order:
            begin = time.perf_counter_ns()
            for _ in range(iterations):
                function()
            elapsed[function] = (
                time.perf_counter_ns() - begin
            ) / iterations / 1000.0
        legacy_samples.append(elapsed[legacy])
        fused_samples.append(elapsed[fused])
        ratios.append(elapsed[fused] / elapsed[legacy])
    paired_ratio = statistics.median(ratios)
    return {
        "legacy_us": statistics.median(legacy_samples),
        "fused_us": statistics.median(fused_samples),
        "paired_ratio_median": paired_ratio,
        "paired_improvement_percent": 100.0 * (1.0 - paired_ratio),
        "legacy_samples_us": legacy_samples,
        "fused_samples_us": fused_samples,
        "paired_ratios": ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--decode-route",
        choices=("default", "legacy", "fused"),
        default="default",
    )
    parser.add_argument(
        "--compare-decode-routes",
        action="store_true",
        help="paired same-process legacy/fused A/B; requires M<4",
    )
    args = parser.parse_args()
    if args.m <= 0 or args.n % 4 or args.k % 32:
        raise ValueError("requires M>0, N%4=0 and K%32=0")

    torch.set_num_threads(args.threads)
    torch.manual_seed(7201)
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    weight = torch.randn((args.n, args.k), dtype=torch.bfloat16)
    qweight, weight_scale = quantize_q4_0(weight)
    rhs = pack_rhs_qsi4c32p(qweight, weight_scale)

    default_fused_decode = set_fused_decode_enabled(False)
    selected_fused_decode = (
        default_fused_decode
        if args.decode_route == "default"
        else args.decode_route == "fused"
    )
    set_fused_decode_enabled(selected_fused_decode)

    def run_direct():
        return linear_w4a8(x, rhs, args.n, args.k)

    def run_legacy_decode():
        set_fused_decode_enabled(False)
        return linear_w4a8(x, rhs, args.n, args.k)

    def run_fused_decode():
        set_fused_decode_enabled(True)
        return linear_w4a8(x, rhs, args.n, args.k)

    def run_production_op():
        return torch.ops.flag_gems.arm_q4_linear(x, rhs, args.n, args.k)

    vllm_slot = _make_vllm_linear(rhs, args.n, args.k)
    unused_weight = torch.empty(0)

    def run_vllm_slot():
        return vllm_slot(x, unused_weight, None)

    direct_output = run_direct()
    production_output = run_production_op()
    slot_output = run_vllm_slot()
    torch.testing.assert_close(production_output, direct_output, rtol=0, atol=0)
    torch.testing.assert_close(slot_output, direct_output, rtol=0, atol=0)
    decode_comparison = None
    if args.compare_decode_routes:
        if args.m >= 4:
            raise ValueError("decode route comparison requires M<4")
        legacy_output = run_legacy_decode()
        fused_output = run_fused_decode()
        torch.testing.assert_close(
            fused_output, legacy_output, rtol=0, atol=0
        )
        decode_comparison = paired_median_us(
            run_legacy_decode,
            run_fused_decode,
            args.warmup,
            args.iters,
            args.batches,
        )
        set_fused_decode_enabled(selected_fused_decode)
    direct_latency = median_us(
        run_direct, args.warmup, args.iters, args.batches
    )
    production_latency = median_us(
        run_production_op, args.warmup, args.iters, args.batches
    )
    slot_latency = median_us(
        run_vllm_slot, args.warmup, args.iters, args.batches
    )
    equivalent_gops = (
        2.0 * args.m * args.n * args.k / production_latency / 1000.0
    )
    result = {
                "status": "PASS",
                "triton_module": triton.__file__,
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "threads": args.threads,
                "decode_route": args.decode_route,
                "output_dtype": str(production_output.dtype),
                "legacy_row_pack": os.getenv(
                    "FLAGGEMS_ARM_Q4_LEGACY_ROW_PACK", "0"
                ).lower()
                in {"1", "true", "on"},
                "m16_as_m8": _use_m8_main_block(),
                "weight_pack_in_timing": False,
                "lhs_pack_in_timing": True,
                "output_allocation_in_timing": True,
                "direct_pipeline_us": direct_latency,
                "production_custom_op_us": production_latency,
                "custom_op_overhead_us": production_latency - direct_latency,
                "vllm_eager_slot_us": slot_latency,
                "eager_slot_overhead_us": slot_latency - direct_latency,
                "equivalent_gops": equivalent_gops,
            }
    if decode_comparison is not None:
        result["decode_route_comparison"] = decode_comparison
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
