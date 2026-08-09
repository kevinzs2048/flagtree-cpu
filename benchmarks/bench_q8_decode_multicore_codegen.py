#!/usr/bin/env python3
"""CIX W8 decode A/B: fused N512 versus shared-quant N64 grid."""

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
    str(ROOT / "python"),
    "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")

import torch  # noqa: E402

from flag_gems.runtime.backend._arm.int8 import (  # noqa: E402
    tle_int8_linear as w8,
)


def paired_median_us(
    first, second, warmup: int, iterations: int, batches: int
) -> tuple[float, float]:
    """Alternate batch order so heterogeneous-core drift affects both arms."""
    for _ in range(warmup):
        first()
        second()
    samples = [[], []]
    functions = (first, second)
    for batch in range(batches):
        order = (0, 1) if batch % 2 == 0 else (1, 0)
        for index in order:
            begin = time.perf_counter_ns()
            for _ in range(iterations):
                functions[index]()
            samples[index].append(
                (time.perf_counter_ns() - begin) / iterations / 1000.0
            )
    return statistics.median(samples[0]), statistics.median(samples[1])


def audit(compiled) -> dict[str, int]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    return {
        "asm_lines": len(assembly.splitlines()),
        "sdot": assembly.count("sdot"),
        "folded_spills": assembly.count("folded spill"),
        "folded_reloads": assembly.count("folded reload"),
        "external_calls": sum(
            " call " in line and "llvm." not in line and " asm " not in line
            for line in llir.splitlines()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=11)
    args = parser.parse_args()
    if args.k % 32 or args.n % 512:
        raise ValueError("requires K%32=0 and N%512=0")

    torch.set_num_threads(args.threads)
    torch.manual_seed(3907)
    x = torch.randn(args.k, dtype=torch.bfloat16)
    weight_kn = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    weight_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    kmajor = w8.pack_weights_sdot(weight_kn)
    packed512 = w8.pack_weights_sdot_blocked(kmajor, 512)
    packed64 = w8.retile_weights_sdot_blocked(packed512, 512, 64)
    q = torch.empty(args.k, dtype=torch.int8)
    x_scale = torch.empty(1, dtype=torch.float32)
    fused_out = torch.empty(args.n, dtype=torch.bfloat16)
    split_out = torch.empty_like(fused_out)

    def run_fused():
        w8._tle_fused_bf16_gemv_kernel[(args.n // 512,)](
            x,
            packed512,
            weight_scale,
            fused_out,
            K=args.k,
            N=args.n,
            BLOCK_N=512,
        )

    def run_split():
        w8._quantize_bf16_w8_rne_kernel[(1,)](
            x, q, x_scale, K=args.k, BLOCK_K=16
        )
        w8._tle_prequant_bf16_gemv_kernel[(args.n // 64,)](
            q,
            x_scale,
            packed64,
            weight_scale,
            split_out,
            K=args.k,
            N=args.n,
            BLOCK_N=64,
        )

    run_fused()
    run_split()
    assert torch.equal(fused_out, split_out)
    fused_compiled = w8._tle_fused_bf16_gemv_kernel.warmup(
        x,
        packed512,
        weight_scale,
        fused_out,
        K=args.k,
        N=args.n,
        BLOCK_N=512,
        grid=(args.n // 512,),
    )
    quant_compiled = w8._quantize_bf16_w8_rne_kernel.warmup(
        x,
        q,
        x_scale,
        K=args.k,
        BLOCK_K=16,
        grid=(1,),
    )
    split_compiled = w8._tle_prequant_bf16_gemv_kernel.warmup(
        q,
        x_scale,
        packed64,
        weight_scale,
        split_out,
        K=args.k,
        N=args.n,
        BLOCK_N=64,
        grid=(args.n // 64,),
    )
    fused_codegen = audit(fused_compiled)
    quant_codegen = audit(quant_compiled)
    split_codegen = audit(split_compiled)
    assert quant_codegen["folded_spills"] == 0
    assert quant_codegen["folded_reloads"] == 0
    assert split_codegen["folded_spills"] == 0
    assert split_codegen["folded_reloads"] == 0
    assert split_codegen["external_calls"] == 0

    fused_us, split_us = paired_median_us(
        run_fused,
        run_split,
        args.warmup,
        args.iters,
        args.batches,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "shape": [1, args.k, args.n],
                "threads": args.threads,
                "bit_exact": True,
                "fused_n512_us": fused_us,
                "split_n64_us": split_us,
                "split_over_fused": split_us / fused_us,
                "speedup": fused_us / split_us,
                "fused_codegen": fused_codegen,
                "split_quant_codegen": quant_codegen,
                "split_matrix_codegen": split_codegen,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
