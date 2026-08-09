#!/usr/bin/env python3
"""Compare complete FlagGems ordinary-JIT and 3.7 AOT W8 Linear."""

from __future__ import annotations

import argparse
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
        "TRITON_CPU_PYTHON",
        ROOT / "ports/triton-cpu-3.7.2/python",
    )
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ["FLAGGEMS_ARM_W8_WHOLE_CODEGEN"] = "1"

import torch  # noqa: E402

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    TLEInt8Linear,
    _quantize_bf16_w8_rne_kernel,
    _w8_decode_sdot_kernel,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(816)
    torch.set_num_threads(1)
    x = torch.randn(args.k, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (args.n, args.k), dtype=torch.int8)
    scale = torch.rand(args.n, dtype=torch.float32) / 127.0

    os.environ["FLAGGEMS_ARM_W8_AOT_BUNDLE"] = args.bundle
    aot_linear = TLEInt8Linear(weight, scale)
    aot_output = aot_linear(x)
    assert aot_linear._ordinary_aot is not None

    os.environ.pop("FLAGGEMS_ARM_W8_AOT_BUNDLE", None)
    jit_linear = TLEInt8Linear(weight, scale)
    jit_output = jit_linear(x)
    assert jit_linear._ordinary_aot is None
    assert torch.equal(aot_output, jit_output)
    quantized = torch.empty(args.k, dtype=torch.int8)
    activation_scale = torch.empty(1, dtype=torch.float32)
    direct_output = torch.empty(args.n, dtype=torch.bfloat16)

    def direct_pipeline() -> None:
        _quantize_bf16_w8_rne_kernel[(1,)](
            x,
            quantized,
            activation_scale,
            K=args.k,
            BLOCK_K=16,
        )
        _w8_decode_sdot_kernel[(1,)](
            quantized,
            activation_scale,
            jit_linear._packed_codegen,
            jit_linear._w_scale_codegen,
            direct_output,
            K=args.k,
            N=jit_linear._N_codegen,
            BLOCK_N=jit_linear._whole_tile_n,
            UNROLL=2,
            WHOLE_PROJECTION=True,
        )

    direct_pipeline()
    assert torch.equal(direct_output[: args.n], aot_output.reshape(-1))

    aot_us = median_us(
        lambda: aot_linear(x), args.warmup, args.iters, args.batches
    )
    jit_us = median_us(
        lambda: jit_linear(x), args.warmup, args.iters, args.batches
    )
    direct_us = median_us(
        direct_pipeline, args.warmup, args.iters, args.batches
    )
    print(
        f"PASS FlagGems BF16-W8 K={args.k} N={args.n}\n"
        f"ordinary_aot_forward_us={aot_us:.3f}\n"
        f"ordinary_jit_forward_us={jit_us:.3f}\n"
        f"preallocated_jit_pipeline_us={direct_us:.3f}\n"
        f"aot_over_jit={aot_us / jit_us:.3f}x\n"
        f"bit_exact={torch.equal(aot_output, jit_output)}"
    )


if __name__ == "__main__":
    main()
