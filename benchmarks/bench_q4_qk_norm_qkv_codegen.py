#!/usr/bin/env python3
"""Paired microbenchmark for Q4 QKV plus Q/K head-RMSNorm fusion."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(ROOT / "ports/triton-cpu-3.7.2/python"),
    "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402
import triton  # noqa: E402

from flag_gems.runtime.backend._arm.fused.patch_qwen3_qk_norm import (  # noqa: E402
    _qk_rms_norm_contiguous_kernel,
)
from flag_gems.runtime.backend._arm.q4.linear import (  # noqa: E402
    linear_w4a8_rmsnorm,
    linear_w4a8_rmsnorm_qk_norm,
    pack_rhs_qsi4c32p,
    quantize_q4_0,
)
from flag_gems.runtime.backend._arm.vector_config import REDUCTION_TILE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--q-elements", type=int, default=2048)
    parser.add_argument("--k-elements", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batches", type=int, default=11)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.n % 4
        or args.k % 32
        or args.q_elements % args.head_dim
        or args.k_elements % args.head_dim
        or args.q_elements + args.k_elements > args.n
    ):
        raise ValueError("invalid Q4 QKV or head layout")
    torch.manual_seed(11003)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    x = torch.randn((1, args.k), dtype=torch.bfloat16)
    input_weight = torch.randn((args.k,), dtype=torch.bfloat16)
    q_weight = torch.randn((args.head_dim,), dtype=torch.bfloat16)
    k_weight = torch.randn((args.head_dim,), dtype=torch.bfloat16)
    qk_weight = torch.cat((q_weight, k_weight)).contiguous()
    q_heads = args.q_elements // args.head_dim
    k_heads = args.k_elements // args.head_dim
    row_weights = torch.cat(
        (
            q_weight.expand(q_heads, -1),
            k_weight.expand(k_heads, -1),
        ),
        dim=0,
    ).contiguous()
    weight = torch.randn((args.n, args.k), dtype=torch.bfloat16)
    qweight, scales = quantize_q4_0(weight)
    rhs = pack_rhs_qsi4c32p(qweight, scales)
    eps = 1.0e-6

    def legacy():
        joined = linear_w4a8_rmsnorm(
            x, input_weight, eps, rhs, args.n, args.k
        )
        qk_output = torch.empty(
            (q_heads + k_heads, args.head_dim), dtype=torch.bfloat16
        )
        _qk_rms_norm_contiguous_kernel[(q_heads + k_heads,)](
            joined,
            row_weights,
            qk_output,
            args.head_dim,
            eps,
            BLOCK_SIZE=REDUCTION_TILE,
            num_warps=1,
            num_stages=1,
        )
        return joined, qk_output

    def fused():
        return linear_w4a8_rmsnorm_qk_norm(
            x,
            input_weight,
            eps,
            qk_weight,
            eps,
            args.q_elements,
            args.k_elements,
            args.head_dim,
            rhs,
            args.n,
            args.k,
        )

    legacy_joined, legacy_qk = legacy()
    fused_joined = fused()
    if not torch.equal(
        fused_joined[:, : args.q_elements + args.k_elements],
        legacy_qk.reshape(1, -1),
    ):
        raise AssertionError("fused Q/K head norms are not bit-exact")
    if not torch.equal(
        fused_joined[:, args.q_elements + args.k_elements :],
        legacy_joined[:, args.q_elements + args.k_elements :],
    ):
        raise AssertionError("fused QKV changed the V projection")

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
        "n": args.n,
        "k": args.k,
        "q_elements": args.q_elements,
        "k_elements": args.k_elements,
        "head_dim": args.head_dim,
        "threads": args.threads,
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
