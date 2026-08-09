#!/usr/bin/env python3
"""Two-stage BF16 W8 decode using only ordinary Triton operations.

Stage one computes the per-vector activation scale and INT8 values.  Stage
two is the packed ``tl.dot`` SDOT kernel from bench_w8a8_codegen.py.  This is
the compiler-only alternative to the dedicated TLE fused/prequant ops.
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from bench_w8a8_codegen import (
    _w8a8_grouped_gemv_kernel,
    median_us,
    pack_w8_microtiles,
)


@triton.jit
def _quantize_bf16_w8_kernel(
    x_ptr,
    x_q_ptr,
    x_scale_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    lanes8 = tl.arange(0, 8)
    absmax = tl.zeros((1,), dtype=tl.float32)
    full_k: tl.constexpr = (K // 32) * 32
    for off in tl.range(0, full_k, 32, loop_unroll_factor=1):
        values0 = tl.load(x_ptr + off + lanes8)
        values1 = tl.load(x_ptr + off + 8 + lanes8)
        values2 = tl.load(x_ptr + off + 16 + lanes8)
        values3 = tl.load(x_ptr + off + 24 + lanes8)
        bits0 = (values0.to(tl.uint16, bitcast=True) & 0x7FFF).to(
            tl.int16
        )
        bits1 = (values1.to(tl.uint16, bitcast=True) & 0x7FFF).to(
            tl.int16
        )
        bits2 = (values2.to(tl.uint16, bitcast=True) & 0x7FFF).to(
            tl.int16
        )
        bits3 = (values3.to(tl.uint16, bitcast=True) & 0x7FFF).to(
            tl.int16
        )
        max01 = tl.where(bits0 > bits1, bits0, bits1)
        max23 = tl.where(bits2 > bits3, bits2, bits3)
        lane_max = tl.where(max01 > max23, max01, max23)
        block_bits = tl.max(lane_max, axis=0).to(tl.uint16)
        block_absmax = (
            block_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        )
        absmax = tl.maximum(absmax, block_absmax)
    if K % 32:
        tail_lanes = tl.arange(0, 32)
        tail_cols = full_k + tail_lanes
        tail_values = tl.load(
            x_ptr + tail_cols, mask=tail_cols < K, other=0.0
        ).to(tl.float32)
        absmax = tl.maximum(
            absmax, tl.max(tl.abs(tail_values), axis=0)
        )

    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inv_scale = 127.0 / absmax
    tl.store(x_scale_ptr + tl.arange(0, 1), scale)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + cols, mask=cols < K, other=0.0).to(tl.float32)
        q = libdevice.rint(x * inv_scale).to(tl.int8)
        tl.store(x_q_ptr + cols, q, mask=cols < K)


def last_compiled(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    return list(cache.values())[-1]


def check_quantizer_finite_edges(x, x_q, x_scale, run_quant) -> None:
    """Check the finite-BF16 ordering assumed by the lane-max graph."""
    normal_x = x.clone()
    edge_bits = torch.tensor(
        [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F80,
            0xBF80,
            0x3F81,
            0xBF81,
            0x7F7F,
            0xFF7F,
        ],
        dtype=torch.uint16,
    )
    repeats = (x.numel() + edge_bits.numel() - 1) // edge_bits.numel()
    x.copy_(edge_bits.repeat(repeats)[: x.numel()].view(torch.bfloat16))
    run_quant()
    edge_absmax = x.float().abs().max().clamp(min=1.0e-8)
    edge_q = (x.float() * (127.0 / edge_absmax)).round().to(torch.int8)
    edge_scale = (edge_absmax / 127.0).reshape(1)
    assert torch.equal(x_q, edge_q)
    assert torch.equal(x_scale, edge_scale)
    x.copy_(normal_x)
    run_quant()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    # 16 maps cleanly to the 128-bit Arm vector reduction.  Wider tiles are
    # scalarized/expanded by the current CPU reduction lowering and produce
    # 1.6k-4.8k assembly lines without improving this decode quantizer.
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--unroll", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()
    if args.k % 4 or args.n % 4:
        raise ValueError("W8 decode requires K%4=0 and N%4=0")

    torch.manual_seed(816)
    x = torch.randn(args.k, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (args.k, args.n), dtype=torch.int8)
    weight_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    packed = pack_w8_microtiles(weight)
    x_q = torch.empty(args.k, dtype=torch.int8)
    x_scale = torch.empty(1, dtype=torch.float32)
    output = torch.empty(args.n, dtype=torch.bfloat16)

    def run_quant() -> None:
        _quantize_bf16_w8_kernel[(1,)](
            x,
            x_q,
            x_scale,
            K=args.k,
            BLOCK_K=args.block_k,
        )

    def run_gemv() -> None:
        _w8a8_grouped_gemv_kernel[(1,)](
            x_q,
            x_scale,
            packed,
            weight_scale,
            output,
            0,
            args.n // 4,
            K=args.k,
            N=args.n,
            UNROLL=args.unroll,
            OUT_BF16=True,
            WHOLE_PROJECTION=True,
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
    assert torch.equal(x_scale, expected_scale.reshape(1))
    assert torch.equal(output, expected)

    check_quantizer_finite_edges(x, x_q, x_scale, run_quant)

    split_us = median_us(run_split, args.warmup, args.iters, args.batches)
    quant_us = median_us(run_quant, args.warmup, args.iters, args.batches)
    gemv_us = median_us(run_gemv, args.warmup, args.iters, args.batches)
    quant_compiled = last_compiled(_quantize_bf16_w8_kernel)
    gemv_compiled = last_compiled(_w8a8_grouped_gemv_kernel)
    quant_asm = quant_compiled.asm["asm"].lower()
    gemv_asm = gemv_compiled.asm["asm"].lower()
    gemv_llir = gemv_compiled.asm["llir"].lower()
    external_calls = [
        line
        for line in gemv_llir.splitlines()
        if " call " in line and "llvm." not in line and " asm " not in line
    ]
    print(
        f"PASS BF16-W8 K={args.k} N={args.n}\n"
        f"split_python_us={split_us:.3f}\n"
        f"quant_python_us={quant_us:.3f}\n"
        f"gemv_python_us={gemv_us:.3f}\n"
        f"bit_exact={torch.equal(output, expected)}\n"
        "finite_bf16_edges_bit_exact=True\n"
        f"quant_asm_lines={len(quant_asm.splitlines())}\n"
        f"gemv_asm_lines={len(gemv_asm.splitlines())}\n"
        f"gemv_asm_sdot={gemv_asm.count('sdot')}\n"
        f"gemv_external_runtime_calls={len(external_calls)}"
    )


if __name__ == "__main__":
    main()
