#!/usr/bin/env python3
"""Sweep ordinary-Triton M=1 W8 ``tl.dot`` output tile widths."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from bench_bf16_w8a8_ordinary_split import (
    _quantize_bf16_w8_kernel,
    check_quantizer_finite_edges,
    last_compiled,
    median_us,
)


@triton.jit
def _w8a8_wide_gemv_kernel(
    x_q_ptr,
    x_scale_ptr,
    packed_ptr,
    weight_scale_ptr,
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
    for block in range(range_begin, range_end):
        dot = tl.zeros((1, BLOCK_N), dtype=tl.int32)
        for group in tl.range(
            0, k_groups, loop_unroll_factor=UNROLL
        ):
            packed_flat = tl.load(
                packed_ptr
                + (block * k_groups + group) * BLOCK_N * 4
                + tl.arange(0, BLOCK_N * 4)
            )
            weight = tl.trans(packed_flat.reshape((BLOCK_N, 4)))
            x = tl.load(
                x_q_ptr + group * 4 + tl.arange(0, 4)
            ).reshape((1, 4))
            dot += tl.dot(x, weight, out_dtype=tl.int32)

        scale = tl.load(weight_scale_ptr + block * BLOCK_N + cols)
        x_scale = tl.load(x_scale_ptr)
        result = dot.to(tl.float32) * x_scale * scale[None, :]
        tl.store(
            out_ptr + block * BLOCK_N + cols,
            result.reshape((BLOCK_N,)).to(tl.bfloat16),
        )


def pack_w8_wide(weight_kn: torch.Tensor, block_n: int) -> torch.Tensor:
    k, n = weight_kn.shape
    if k % 4 or n % block_n or block_n % 4:
        raise ValueError("wide W8 pack requires K%4=0 and N%BLOCK_N=0")
    return (
        weight_kn.reshape(k // 4, 4, n // block_n, block_n)
        .permute(2, 0, 3, 1)
        .contiguous()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--unroll", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(864)
    x = torch.randn(args.k, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (args.k, args.n), dtype=torch.int8)
    weight_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    packed = pack_w8_wide(weight, args.block_n)
    x_q = torch.empty(args.k, dtype=torch.int8)
    x_scale = torch.empty(1, dtype=torch.float32)
    output = torch.empty(args.n, dtype=torch.bfloat16)

    def run_quant() -> None:
        _quantize_bf16_w8_kernel[(1,)](
            x, x_q, x_scale, K=args.k, BLOCK_K=16
        )

    def run_gemv() -> None:
        _w8a8_wide_gemv_kernel[(1,)](
            x_q,
            x_scale,
            packed,
            weight_scale,
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
        run_gemv()

    run_split()
    x_fp32 = x.float()
    expected_scale = x_fp32.abs().max().clamp(min=1.0e-8) / 127.0
    expected_q = (
        (x_fp32 * (127.0 / x_fp32.abs().max().clamp(min=1.0e-8)))
        .clamp(-128, 127)
        .round()
        .to(torch.int8)
    )
    expected = (
        (expected_q.to(torch.int32) @ weight.to(torch.int32)).float()
        * expected_scale
        * weight_scale
    ).to(torch.bfloat16)
    assert torch.equal(x_q, expected_q)
    assert torch.equal(output, expected)
    check_quantizer_finite_edges(x, x_q, x_scale, run_quant)

    split_us = median_us(run_split, args.warmup, args.iters, args.batches)
    gemv_us = median_us(run_gemv, args.warmup, args.iters, args.batches)
    compiled = last_compiled(_w8a8_wide_gemv_kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    tttcir = compiled.asm["tttcir"].lower()
    packed_load_signature = (
        f"memref<{args.block_n * 4}xi8>, vector<16xi8>"
    )
    packed_direct_loads = tttcir.count(packed_load_signature)
    logical_weight_tmp = f"memref<4x{args.block_n}xi8>" in tttcir
    assert packed_direct_loads == args.unroll
    assert not logical_weight_tmp
    print(
        f"PASS BF16-W8-wide K={args.k} N={args.n} BN={args.block_n}\n"
        f"split_python_us={split_us:.3f}\n"
        f"gemv_python_us={gemv_us:.3f}\n"
        f"bit_exact={torch.equal(output, expected)}\n"
        "finite_bf16_edges_bit_exact=True\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"stack_load_store={sum('[sp' in line and line.lstrip().startswith(('ld', 'st')) for line in assembly.splitlines())}\n"
        f"llir_alloca_i32={sum('alloca' in line and 'i32' in line for line in llir.splitlines())}\n"
        f"packed_direct_loads={packed_direct_loads}\n"
        f"logical_weight_tmp={logical_weight_tmp}"
    )


if __name__ == "__main__":
    main()
