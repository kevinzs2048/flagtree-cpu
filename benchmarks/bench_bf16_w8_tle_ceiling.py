#!/usr/bin/env python3
"""Measure the existing one-launch TLE BF16-W8 whole-projection ceiling."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/home/kevin/triton-opt-cpu")
VENV_SITE = Path("/home/cix/venv-fep-e2e/lib/python3.11/site-packages")
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(ROOT / "python"),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ["FLAGGEMS_ARM_W8_WHOLE_CODEGEN"] = "1"

import torch  # noqa: E402

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    TLEInt8Linear,
    _tle_whole_bf16_gemv_kernel,
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
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(816)
    torch.set_num_threads(1)
    x = torch.randn(args.k, dtype=torch.bfloat16)
    weight_nk = torch.randint(
        -127, 128, (args.n, args.k), dtype=torch.int8
    )
    weight_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    linear = TLEInt8Linear(weight_nk, weight_scale)
    expected = linear(x)
    output = torch.empty(linear._N_codegen, dtype=torch.bfloat16)

    def run() -> None:
        _tle_whole_bf16_gemv_kernel[(1,)](
            x,
            linear._packed_codegen,
            linear._w_scale_codegen,
            output,
            K=linear.K,
            N=linear._N_codegen,
            TILE_N=linear._whole_tile_n,
        )

    run()
    assert torch.equal(output[: args.n], expected.reshape(-1))
    latency = median_us(run, args.warmup, args.iters, args.batches)
    cache = next(
        iter(_tle_whole_bf16_gemv_kernel.device_caches.values())
    )[0]
    compiled = list(cache.values())[-1]
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    external_calls = [
        line
        for line in llir.splitlines()
        if " call " in line and "llvm." not in line
    ]
    print(
        f"PASS TLE BF16-W8 K={args.k} N={args.n}\n"
        f"one_python_launch_us={latency:.3f}\n"
        f"bit_exact={torch.equal(output[:args.n], expected.reshape(-1))}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"external_runtime_calls={len(external_calls)}"
    )


if __name__ == "__main__":
    main()
