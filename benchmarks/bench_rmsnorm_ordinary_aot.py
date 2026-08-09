#!/usr/bin/env python3
"""Shape-specialized ordinary-Triton BF16 RMSNorm for direct AOT calls."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from bench_bf16_w8a8_ordinary_split import last_compiled, median_us


@triton.jit
def _rms_norm_aot_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + row * N + cols).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / N + EPS)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + row * N + cols).to(tl.float32)
        weight = tl.load(weight_ptr + cols)
        normalized = (x * rrms).to(tl.bfloat16)
        tl.store(
            out_ptr + row * N + cols,
            (normalized * weight).to(tl.bfloat16),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(867)
    x = torch.randn(args.m, args.n, dtype=torch.bfloat16)
    weight = torch.randn(args.n, dtype=torch.bfloat16)
    output = torch.empty_like(x)

    def run() -> None:
        _rms_norm_aot_kernel[(args.m,)](
            x,
            weight,
            output,
            M=args.m,
            N=args.n,
            EPS=args.eps,
            BLOCK_N=args.block_n,
        )

    run()
    xf = x.float()
    rrms = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) / args.n + args.eps)
    expected = ((xf * rrms).to(torch.bfloat16) * weight).to(torch.bfloat16)
    assert torch.equal(output, expected)

    latency = median_us(run, args.warmup, args.iters, args.batches)
    compiled = last_compiled(_rms_norm_aot_kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    print(
        f"PASS ordinary RMSNorm M={args.m} N={args.n}\n"
        f"python_launch_us={latency:.3f}\n"
        f"bit_exact={torch.equal(output, expected)}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"stack_load_store={sum('[sp' in line and line.lstrip().startswith(('ld', 'st')) for line in assembly.splitlines())}\n"
        f"external_calls={sum(' call ' in line and 'llvm.' not in line for line in llir.splitlines())}"
    )


if __name__ == "__main__":
    main()
