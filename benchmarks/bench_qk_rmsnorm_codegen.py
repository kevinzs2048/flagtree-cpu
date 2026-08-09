#!/usr/bin/env python3
"""Microbenchmark independent versus fused Q/K BF16 RMSNorm."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/home/kevin/triton-opt-cpu")
VENV_SITE = Path("/home/cix/venv-fep-e2e/lib/python3.11/site-packages")
TRITON_PYTHON = Path(
    os.getenv("TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python")
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402

from flag_gems.runtime.backend._arm.fused.patch_qwen3_qk_norm import (  # noqa: E402
    _qk_rms_norm_contiguous_kernel,
)
from flag_gems.runtime.backend._arm.fused.patch_qwen3_rmsnorm import (  # noqa: E402
    _rms_norm,
)
from flag_gems.runtime.backend._arm.vector_config import (  # noqa: E402
    REDUCTION_TILE,
)


def median_us(fn, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        begin = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - begin) / 1e3)
        assert result is not None
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=301)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(808)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    qk = torch.randn(
        (args.q_heads + args.kv_heads, args.head_dim),
        dtype=torch.bfloat16,
    )
    q = qk[: args.q_heads].reshape(1, 1, args.q_heads, args.head_dim)
    k = qk[args.q_heads :].reshape(
        1, 1, args.kv_heads, args.head_dim
    )
    qw = torch.randn(args.head_dim, dtype=torch.bfloat16)
    kw = torch.randn(args.head_dim, dtype=torch.bfloat16)
    qk_row_weight = torch.cat(
        (
            qw.expand(args.q_heads, -1),
            kw.expand(args.kv_heads, -1),
        ),
        dim=0,
    ).contiguous()
    eps = 1e-6

    if args.compile_only:
        output = torch.empty_like(qk)
        compiled = _qk_rms_norm_contiguous_kernel.warmup(
            qk,
            qk_row_weight,
            output,
            args.head_dim,
            eps,
            BLOCK_SIZE=REDUCTION_TILE,
            grid=(args.q_heads + args.kv_heads,),
        )
        assembly = compiled.asm["asm"].lower()
        llir = compiled.asm["llir"].lower()
        stack_load_store = sum(
            "[sp" in line and line.lstrip().startswith(("ld", "st"))
            for line in assembly.splitlines()
        )
        external_calls = sum(
            " call " in line and "llvm." not in line and " asm " not in line
            for line in llir.splitlines()
        )
        if len(assembly.splitlines()) >= 300 or external_calls:
            raise RuntimeError(
                "Q/K RMSNorm codegen regression: "
                f"asm_lines={len(assembly.splitlines())} "
                f"external_calls={external_calls}"
            )
        print(
            f"COMPILED fused Q/K RMSNorm Hq={args.q_heads} "
            f"Hkv={args.kv_heads} D={args.head_dim}\n"
            f"asm_lines={len(assembly.splitlines())}\n"
            f"stack_load_store={stack_load_store}\n"
            f"external_calls={external_calls}"
        )
        return

    expected_q = _rms_norm(q, qw, eps)
    expected_k = _rms_norm(k, kw, eps)

    def fused():
        output = torch.empty_like(qk)
        _qk_rms_norm_contiguous_kernel[
            (args.q_heads + args.kv_heads,)
        ](
            qk,
            qk_row_weight,
            output,
            args.head_dim,
            eps,
            BLOCK_SIZE=REDUCTION_TILE,
            num_warps=1,
            num_stages=1,
        )
        return (
            output[: args.q_heads].reshape(q.shape),
            output[args.q_heads :].reshape(k.shape),
        )

    actual_q, actual_k = fused()
    exact = torch.equal(expected_q, actual_q) and torch.equal(
        expected_k, actual_k
    )

    def independent():
        return _rms_norm(q, qw, eps), _rms_norm(k, kw, eps)

    for _ in range(20):
        independent()
        fused()
    independent_us = median_us(independent, args.iterations)
    fused_us = median_us(fused, args.iterations)
    print(
        f"threads={args.threads}\n"
        f"independent_us={independent_us:.3f}\n"
        f"fused_us={fused_us:.3f}\n"
        f"fused_over_independent={fused_us / independent_us:.3f}x\n"
        f"bit_exact={exact}"
    )


if __name__ == "__main__":
    main()
