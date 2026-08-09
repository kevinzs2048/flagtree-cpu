#!/usr/bin/env python3
"""Paired microbenchmark for ordinary-Triton RMSNorm + Q4 QKV fusion."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = Path("/home/cix/venv-fep-e2e/lib/python3.11/site-packages")
TRITON_PYTHON = ROOT / "ports/triton-cpu-3.7.2/python"
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402
import triton  # noqa: E402

from flag_gems.runtime.backend._arm.fused.patch_qwen3_rmsnorm import (  # noqa: E402
    _rms_norm,
)
from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (  # noqa: E402
    fused_add_rms_norm,
)
from flag_gems.runtime.backend._arm.q4.linear import (  # noqa: E402
    linear_w4a8,
    linear_w4a8_add_rmsnorm,
    linear_w4a8_rmsnorm,
    pack_rhs_qsi4c32p,
    quantize_q4_0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--add-residual", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batches", type=int, default=11)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.m < 4 or args.n % 4 or args.k % 32:
        raise ValueError("requires M=1..3, N%4=0 and K%32=0")
    torch.manual_seed(9713)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    rms_weight = torch.randn((args.k,), dtype=torch.bfloat16)
    weight = torch.randn((args.n, args.k), dtype=torch.bfloat16)
    qweight, scales = quantize_q4_0(weight)
    rhs = pack_rhs_qsi4c32p(qweight, scales)
    eps = 1.0e-6

    if args.add_residual:
        legacy_x = torch.empty_like(x)
        legacy_residual = torch.empty_like(residual)
        fused_x = torch.empty_like(x)
        fused_residual = torch.empty_like(residual)

        def legacy():
            legacy_x.copy_(x)
            legacy_residual.copy_(residual)
            normalized, updated = fused_add_rms_norm(
                legacy_x,
                legacy_residual,
                (args.k,),
                rms_weight,
                eps,
            )
            return linear_w4a8(
                normalized, rhs, args.n, args.k
            ), updated

        def fused():
            fused_x.copy_(x)
            fused_residual.copy_(residual)
            return linear_w4a8_add_rmsnorm(
                fused_x,
                fused_residual,
                rms_weight,
                eps,
                rhs,
                args.n,
                args.k,
            )

        legacy_output, legacy_updated = legacy()
        fused_output, fused_updated = fused()
        if not torch.equal(fused_updated, legacy_updated):
            raise AssertionError("fused updated residual is not bit-exact")
    else:
        def legacy():
            normalized = _rms_norm(x, rms_weight, eps)
            return linear_w4a8(normalized, rhs, args.n, args.k)

        def fused():
            return linear_w4a8_rmsnorm(
                x, rms_weight, eps, rhs, args.n, args.k
            )

        legacy_output = legacy()
        fused_output = fused()
    if not torch.equal(fused_output, legacy_output):
        raise AssertionError("fused RMSNorm/QKV output is not bit-exact")
    for function in (legacy, fused):
        for _ in range(args.warmup):
            function()

    samples = {"legacy_us": [], "fused_us": []}
    ratios = []
    for batch in range(args.batches):
        order = (legacy, fused) if batch % 2 == 0 else (fused, legacy)
        elapsed = {}
        for function in order:
            begin = time.perf_counter_ns()
            for _ in range(args.iters):
                function()
            elapsed[function] = (
                time.perf_counter_ns() - begin
            ) / args.iters / 1000.0
        samples["legacy_us"].append(elapsed[legacy])
        samples["fused_us"].append(elapsed[fused])
        ratios.append(elapsed[fused] / elapsed[legacy])

    ratio = statistics.median(ratios)
    result = {
        "status": "PASS",
        "triton": triton.__file__,
        "target": str(triton.runtime.driver.active.get_current_target()),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "threads": args.threads,
        "add_residual": args.add_residual,
        "bit_exact": True,
        "legacy_median_us": statistics.median(samples["legacy_us"]),
        "fused_median_us": statistics.median(samples["fused_us"]),
        "paired_ratio_median": ratio,
        "paired_improvement_percent": 100.0 * (1.0 - ratio),
        "samples": samples,
        "paired_ratios": ratios,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)


if __name__ == "__main__":
    main()
