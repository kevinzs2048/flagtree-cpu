#!/usr/bin/env python3
"""A/B wide and register-bounded Q8 activation quantizer specializations."""

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
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(ROOT / "python"),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    _quantize_rows_bf16_kernel,
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


def audit(compiled) -> dict[str, int]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    return {
        "asm_lines": len(assembly.splitlines()),
        "folded_spills": assembly.count("folded spill"),
        "folded_reloads": assembly.count("folded reload"),
        "external_calls": sum(
            " call " in line and "llvm." not in line and " asm " not in line
            for line in llir.splitlines()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batches", type=int, default=11)
    args = parser.parse_args()
    if args.m <= 0 or args.k % 128:
        raise ValueError("requires M>0 and K%128=0")

    torch.set_num_threads(1)
    torch.manual_seed(8311)
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    output = {}

    def make_variant(block_k: int):
        quantized = torch.empty_like(x, dtype=torch.int8)
        scale = torch.empty(args.m, dtype=torch.float32)

        def run():
            _quantize_rows_bf16_kernel[(args.m,)](
                x,
                quantized,
                scale,
                args.m,
                args.k,
                x.stride(0),
                x.stride(1),
                quantized.stride(0),
                quantized.stride(1),
                BLOCK_K=block_k,
                num_warps=1,
                num_stages=1,
            )

        run()
        compiled = _quantize_rows_bf16_kernel.warmup(
            x,
            quantized,
            scale,
            args.m,
            args.k,
            x.stride(0),
            x.stride(1),
            quantized.stride(0),
            quantized.stride(1),
            BLOCK_K=block_k,
            num_warps=1,
            num_stages=1,
            grid=(args.m,),
        )
        latency = median_us(run, args.warmup, args.iters, args.batches)
        return quantized, scale, latency, audit(compiled)

    wide_q, wide_scale, wide_us, wide_codegen = make_variant(128)
    bounded_q, bounded_scale, bounded_us, bounded_codegen = make_variant(16)
    assert torch.equal(wide_q, bounded_q)
    assert torch.equal(wide_scale, bounded_scale)
    output.update(
        {
            "status": "PASS",
            "m": args.m,
            "k": args.k,
            "bit_exact": True,
            "wide_k128_us": wide_us,
            "bounded_k16_us": bounded_us,
            "bounded_over_wide": bounded_us / wide_us,
            "wide_codegen": wide_codegen,
            "bounded_codegen": bounded_codegen,
        }
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
