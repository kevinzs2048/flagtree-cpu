#!/usr/bin/env python3
"""Ordinary ``tl.dot`` BF16-W8 gate/up SwiGLU experiment."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from bench_bf16_w8a8_ordinary_split import (
    _quantize_bf16_w8_kernel,
    last_compiled,
    median_us,
)
from bench_bf16_w8a8_wide_split import pack_w8_wide


@triton.jit
def _w8_swiglu_wide_kernel(
    x_q_ptr,
    x_scale_ptr,
    gate_packed_ptr,
    up_packed_ptr,
    gate_scale_ptr,
    up_scale_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    k_groups: tl.constexpr = K // 4
    cols = tl.arange(0, BLOCK_N)
    packed_offsets = tl.arange(0, BLOCK_N * 4)
    for block in range(range_begin, range_end):
        gate_dot = tl.zeros((1, BLOCK_N), dtype=tl.int32)
        up_dot = tl.zeros((1, BLOCK_N), dtype=tl.int32)
        for group in tl.range(
            0, k_groups, loop_unroll_factor=UNROLL
        ):
            packed_base = (block * k_groups + group) * BLOCK_N * 4
            gate_flat = tl.load(gate_packed_ptr + packed_base + packed_offsets)
            up_flat = tl.load(up_packed_ptr + packed_base + packed_offsets)
            gate_weight = tl.trans(gate_flat.reshape((BLOCK_N, 4)))
            up_weight = tl.trans(up_flat.reshape((BLOCK_N, 4)))
            x = tl.load(
                x_q_ptr + group * 4 + tl.arange(0, 4)
            ).reshape((1, 4))
            gate_dot += tl.dot(x, gate_weight, out_dtype=tl.int32)
            up_dot += tl.dot(x, up_weight, out_dtype=tl.int32)

        output_offset = block * BLOCK_N + cols
        x_scale = tl.load(x_scale_ptr)
        gate_scale = tl.load(gate_scale_ptr + output_offset)
        up_scale = tl.load(up_scale_ptr + output_offset)
        gate = gate_dot.to(tl.float32) * x_scale * gate_scale[None, :]
        up = up_dot.to(tl.float32) * x_scale * up_scale[None, :]
        # Match the model-visible BF16 boundaries of two W8 Linear outputs
        # followed by BF16 SiLU and multiply.
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        silu = gate / (1.0 + tl.exp(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        result = (silu * up).to(tl.bfloat16)
        tl.store(out_ptr + output_offset, result.reshape((BLOCK_N,)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--unroll", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(865)
    x = torch.randn(args.k, dtype=torch.bfloat16) * 0.2
    gate_weight = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    up_weight = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    gate_packed = pack_w8_wide(gate_weight, args.block_n)
    up_packed = pack_w8_wide(up_weight, args.block_n)
    gate_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    up_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    x_q = torch.empty(args.k, dtype=torch.int8)
    x_scale = torch.empty(1, dtype=torch.float32)
    output = torch.empty(args.n, dtype=torch.bfloat16)

    def run_quant() -> None:
        _quantize_bf16_w8_kernel[(1,)](
            x, x_q, x_scale, K=args.k, BLOCK_K=16
        )

    def run_swiglu() -> None:
        _w8_swiglu_wide_kernel[(1,)](
            x_q,
            x_scale,
            gate_packed,
            up_packed,
            gate_scale,
            up_scale,
            output,
            0,
            args.n // args.block_n,
            K=args.k,
            N=args.n,
            BLOCK_N=args.block_n,
            UNROLL=args.unroll,
        )

    def run_split() -> None:
        run_quant()
        run_swiglu()

    run_split()
    xf = x.float()
    absmax = xf.abs().max().clamp(min=1.0e-8)
    expected_scale = absmax / 127.0
    expected_q = (
        (xf * (127.0 / absmax)).clamp(-128, 127).round().to(torch.int8)
    )
    gate = (
        (expected_q.to(torch.int32) @ gate_weight.to(torch.int32)).float()
        * expected_scale
        * gate_scale
    ).to(torch.bfloat16)
    up = (
        (expected_q.to(torch.int32) @ up_weight.to(torch.int32)).float()
        * expected_scale
        * up_scale
    ).to(torch.bfloat16)
    expected = (F.silu(gate) * up).to(torch.bfloat16)
    assert torch.equal(x_q, expected_q)
    assert torch.equal(output, expected)

    split_us = median_us(run_split, args.warmup, args.iters, args.batches)
    swiglu_us = median_us(run_swiglu, args.warmup, args.iters, args.batches)
    compiled = last_compiled(_w8_swiglu_wide_kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    print(
        f"PASS ordinary BF16-W8 SwiGLU K={args.k} N={args.n} "
        f"BN={args.block_n}\n"
        f"split_python_us={split_us:.3f}\n"
        f"swiglu_python_us={swiglu_us:.3f}\n"
        f"bit_exact={torch.equal(output, expected)}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"stack_load_store={sum('[sp' in line and line.lstrip().startswith(('ld', 'st')) for line in assembly.splitlines())}\n"
        f"llir_alloca_i32={sum('alloca' in line and 'i32' in line for line in llir.splitlines())}"
    )


if __name__ == "__main__":
    main()
