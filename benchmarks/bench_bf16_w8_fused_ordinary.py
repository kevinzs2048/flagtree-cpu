#!/usr/bin/env python3
"""Test one-kernel ordinary-Triton BF16-to-W8 decode against the split path."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    _quantize_bf16_w8_rne_kernel,
    _w8_decode_sdot_kernel,
    pack_weights_sdot,
    pack_weights_sdot_blocked,
    retile_weights_sdot_blocked,
)


@triton.jit
def _bf16_w8_fused_requant_sdot_kernel(
    x_ptr,
    packed_ptr,
    weight_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Quantize in each N block to avoid a scratch buffer and second launch."""
    lanes8 = tl.arange(0, 8)
    absmax = tl.zeros((1,), dtype=tl.float32)
    for off in tl.range(0, K, 32, loop_unroll_factor=1):
        values0 = tl.load(x_ptr + off + lanes8)
        values1 = tl.load(x_ptr + off + 8 + lanes8)
        values2 = tl.load(x_ptr + off + 16 + lanes8)
        values3 = tl.load(x_ptr + off + 24 + lanes8)
        bits0 = (values0.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        bits1 = (values1.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        bits2 = (values2.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        bits3 = (values3.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        max01 = tl.where(bits0 > bits1, bits0, bits1)
        max23 = tl.where(bits2 > bits3, bits2, bits3)
        lane_max = tl.where(max01 > max23, max01, max23)
        block_bits = tl.max(lane_max, axis=0).to(tl.uint16)
        block_absmax = block_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        absmax = tl.maximum(absmax, block_absmax)

    absmax = tl.maximum(absmax, 1.0e-8)
    activation_scale = absmax / 127.0
    inverse_scale = 127.0 / absmax
    k_groups: tl.constexpr = K // 4
    cols = tl.arange(0, BLOCK_N)
    for block in range(0, N // BLOCK_N):
        accumulator = tl.zeros((1, BLOCK_N), dtype=tl.int32)
        for group in tl.range(
            0, k_groups, loop_unroll_factor=UNROLL
        ):
            values = tl.load(
                x_ptr + group * 4 + tl.arange(0, 4)
            ).to(tl.float32)
            activation = libdevice.rint(values * inverse_scale).to(
                tl.int8
            ).reshape((1, 4))
            packed_flat = tl.load(
                packed_ptr
                + (block * k_groups + group) * BLOCK_N * 4
                + tl.arange(0, BLOCK_N * 4)
            )
            weight = tl.trans(packed_flat.reshape((BLOCK_N, 4)))
            accumulator += tl.dot(
                activation, weight, out_dtype=tl.int32
            )

        weight_scale = tl.load(
            weight_scale_ptr + block * BLOCK_N + cols
        )
        result = (
            accumulator.to(tl.float32)
            * activation_scale
            * weight_scale[None, :]
        )
        tl.store(
            out_ptr + block * BLOCK_N + cols,
            result.reshape((BLOCK_N,)).to(tl.bfloat16),
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


def last_compiled(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    return list(cache.values())[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batches", type=int, default=9)
    args = parser.parse_args()
    if args.k % 32 or args.n % args.block_n:
        raise ValueError("requires K%32=0 and N%BLOCK_N=0")

    torch.manual_seed(868)
    torch.set_num_threads(1)
    x = torch.randn(args.k, dtype=torch.bfloat16)
    weight = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    weight_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    packed = pack_weights_sdot_blocked(
        pack_weights_sdot(weight), min(512, args.n)
    )
    packed = retile_weights_sdot_blocked(
        packed, min(512, args.n), args.block_n
    )
    quantized = torch.empty(args.k, dtype=torch.int8)
    activation_scale = torch.empty(1, dtype=torch.float32)
    split_output = torch.empty(args.n, dtype=torch.bfloat16)
    fused_output = torch.empty_like(split_output)

    def run_split() -> None:
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
            packed,
            weight_scale,
            split_output,
            K=args.k,
            N=args.n,
            BLOCK_N=args.block_n,
            UNROLL=2,
            WHOLE_PROJECTION=True,
        )

    def run_fused() -> None:
        _bf16_w8_fused_requant_sdot_kernel[(1,)](
            x,
            packed,
            weight_scale,
            fused_output,
            K=args.k,
            N=args.n,
            BLOCK_N=args.block_n,
            UNROLL=2,
        )

    run_split()
    run_fused()
    if not torch.equal(fused_output, split_output):
        delta = fused_output.float() - split_output.float()
        raise AssertionError(
            f"mismatch count={torch.count_nonzero(delta).item()} "
            f"max_abs={delta.abs().max().item()}"
        )

    split_us = median_us(run_split, args.warmup, args.iters, args.batches)
    fused_us = median_us(run_fused, args.warmup, args.iters, args.batches)
    compiled = last_compiled(_bf16_w8_fused_requant_sdot_kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    calls = [
        line
        for line in llir.splitlines()
        if " call " in line and "llvm." not in line and " asm " not in line
    ]
    print(
        f"PASS fused ordinary BF16-W8 K={args.k} N={args.n} "
        f"BN={args.block_n}\n"
        f"split_python_us={split_us:.3f}\n"
        f"fused_python_us={fused_us:.3f}\n"
        f"fused_over_split={fused_us / split_us:.3f}x\n"
        f"bit_exact={torch.equal(fused_output, split_output)}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"folded_spills={assembly.count('folded spill')}\n"
        f"folded_reloads={assembly.count('folded reload')}\n"
        f"external_compute_calls={len(calls)}\n"
        f"residual_dot={'triton_cpu.dot' in llir}"
    )


if __name__ == "__main__":
    main()
