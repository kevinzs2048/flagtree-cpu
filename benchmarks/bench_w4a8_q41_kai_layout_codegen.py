#!/usr/bin/env python3
"""Ordinary Triton Q4_1 x Q8_1 decode over a KAI-style N4/K32 layout.

This is not a KleidiAI-defined Q4_1 format.  It extends the same 4-output,
32-K physical microtile used by KAI's Q4_0 decode kernel with four FP16
minimum values.  The compute loop remains ordinary Triton operations.
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl

from bench_w4_kleidiai_layout_codegen import audit_codegen, last_compiled
from bench_w4a8_codegen import median_us


@triton.jit
def _kai_q41_layout_split_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    groups: tl.constexpr = K // 32
    lhs_group_stride: tl.constexpr = 36
    rhs_group_stride: tl.constexpr = 80
    rhs_tile_stride: tl.constexpr = groups * rhs_group_stride
    q_lanes = tl.arange(0, 16)
    x_lanes = tl.arange(0, 8)
    output_lanes = tl.arange(0, 4)
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    for tile in range(range_begin, range_end):
        result = tl.zeros((4,), dtype=tl.float32)
        lhs_group_ptr = lhs_packed_ptr
        rhs_group_ptr = rhs_packed_ptr + tile * rhs_tile_stride
        for group in tl.range(0, groups, loop_unroll_factor=UNROLL):
            lhs_scale = tl.load(
                lhs_group_ptr.to(tl.pointer_type(tl.float16))
            ).to(tl.float32)
            lhs_sum = tl.load(
                (lhs_group_ptr + 2).to(tl.pointer_type(tl.float16))
            ).to(tl.float32)
            rhs_half_ptr = rhs_group_ptr.to(tl.pointer_type(tl.float16))
            rhs_scale = tl.load(rhs_half_ptr + output_lanes).to(tl.float32)
            rhs_minimum = tl.load(rhs_half_ptr + 4 + output_lanes).to(tl.float32)

            q0 = tl.load(rhs_group_ptr + 16 + q_lanes)
            q1 = tl.load(rhs_group_ptr + 32 + q_lanes)
            q2 = tl.load(rhs_group_ptr + 48 + q_lanes)
            q3 = tl.load(rhs_group_ptr + 64 + q_lanes)
            q0_low = (q0 << 4).to(tl.int8).reshape((4, 4))
            q1_low = (q1 << 4).to(tl.int8).reshape((4, 4))
            q2_low = (q2 << 4).to(tl.int8).reshape((4, 4))
            q3_low = (q3 << 4).to(tl.int8).reshape((4, 4))
            q0_high = (q0 & 0xF0).to(tl.int8).reshape((4, 4))
            q1_high = (q1 & 0xF0).to(tl.int8).reshape((4, 4))
            q2_high = (q2 & 0xF0).to(tl.int8).reshape((4, 4))
            q3_high = (q3 & 0xF0).to(tl.int8).reshape((4, 4))

            x_ptr = lhs_group_ptr + 4
            x0 = tl.load(x_ptr + x_lanes).to(tl.int8).reshape((2, 4))
            x1 = tl.load(x_ptr + 8 + x_lanes).to(tl.int8).reshape((2, 4))
            x2 = tl.load(x_ptr + 16 + x_lanes).to(tl.int8).reshape((2, 4))
            x3 = tl.load(x_ptr + 24 + x_lanes).to(tl.int8).reshape((2, 4))
            x0 = tl.join(x0.reshape((8,)), x0.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x1 = tl.join(x1.reshape((8,)), x1.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x2 = tl.join(x2.reshape((8,)), x2.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x3 = tl.join(x3.reshape((8,)), x3.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))

            partial01 = tl.sum(
                q0_low.to(tl.int32) * x0.to(tl.int32), axis=1
            )
            partial23 = tl.sum(
                q1_low.to(tl.int32) * x0.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q2_low.to(tl.int32) * x1.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q3_low.to(tl.int32) * x1.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q0_high.to(tl.int32) * x2.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q1_high.to(tl.int32) * x2.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q2_high.to(tl.int32) * x3.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q3_high.to(tl.int32) * x3.to(tl.int32), axis=1
            )

            partial = tl.join(partial01, partial23).permute(
                1, 0
            ).reshape((4, 2))
            dot_scaled16 = tl.sum(partial, axis=1)
            dot = dot_scaled16.to(tl.float32) * (1.0 / 16.0)
            result += dot * (lhs_scale * rhs_scale)
            result += (rhs_minimum + 8.0 * rhs_scale) * lhs_sum
            lhs_group_ptr += lhs_group_stride
            rhs_group_ptr += rhs_group_stride

        result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
        tl.store(out_ptr + tile * 4 + output_lanes, result)


def pack_lhs(x: torch.Tensor, scale: torch.Tensor, scaled_sum: torch.Tensor) -> torch.Tensor:
    groups = x.numel() // 32
    packed = torch.empty((groups, 36), dtype=torch.uint8)
    packed[:, :2].view(torch.float16).copy_(scale.reshape(groups, 1))
    packed[:, 2:4].view(torch.float16).copy_(scaled_sum.reshape(groups, 1))
    packed[:, 4:].view(torch.int8).copy_(x.reshape(groups, 32))
    return packed.reshape(-1)


def pack_rhs(q: torch.Tensor, scale: torch.Tensor, minimum: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    groups = k // 32
    tiles = n // 4
    packed = torch.empty((tiles, groups, 80), dtype=torch.uint8)
    packed[:, :, :8].view(torch.float16).copy_(
        scale.reshape(groups, tiles, 4).permute(1, 0, 2)
    )
    packed[:, :, 8:16].view(torch.float16).copy_(
        minimum.reshape(groups, tiles, 4).permute(1, 0, 2)
    )
    for tile in range(tiles):
        for group in range(groups):
            block = q[tile * 4 : tile * 4 + 4, group * 32 : group * 32 + 32]
            vectors = []
            for k_begin, outputs in (
                (0, (0, 1)), (0, (2, 3)), (8, (0, 1)), (8, (2, 3))
            ):
                lo = block[list(outputs), k_begin : k_begin + 8].to(torch.int16)
                hi = block[list(outputs), k_begin + 16 : k_begin + 24].to(torch.int16)
                vectors.append(((lo | (hi << 4)) ^ 0x88).to(torch.uint8).reshape(-1))
            packed[tile, group, 16:].copy_(torch.cat(vectors))
    return packed.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=9728)
    parser.add_argument("--n", type=int, default=2560)
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()
    if args.k % 32 or args.n % 4:
        raise ValueError("requires K%32=0 and N%4=0")

    clamp = torch.tensor([-math.inf, math.inf], dtype=torch.float32)
    if args.compile_only:
        byte_buffer = torch.empty(1, dtype=torch.uint8)
        output = torch.empty(1, dtype=torch.float32)
        compiled = _kai_q41_layout_split_kernel.warmup(
            byte_buffer,
            byte_buffer,
            clamp,
            output,
            0,
            args.n // 4,
            K=args.k,
            N=args.n,
            UNROLL=args.unroll,
            grid=(1,),
        )
        assembly, _, sdot, addp, stack, calls = audit_codegen(compiled, args.unroll)
        print(
            f"COMPILED KAI-style Q4_1 K={args.k} N={args.n} UNROLL={args.unroll}\n"
            f"asm_lines={len(assembly.splitlines())}\nasm_sdot={sdot}\n"
            f"asm_addp={addp}\nstack_load_store={stack}\nexternal_calls={calls}"
        )
        return

    torch.manual_seed(4141)
    x = torch.randint(-127, 128, (args.k,), dtype=torch.int8)
    q = torch.randint(0, 16, (args.n, args.k), dtype=torch.uint8)
    lhs_scale = torch.rand(args.k // 32, dtype=torch.float16) * 0.02 + 0.001
    lhs_sum = (
        x.reshape(-1, 32).float().sum(1) * lhs_scale.float()
    ).to(torch.float16)
    rhs_scale = torch.rand(args.k // 32, args.n, dtype=torch.float16) * 0.02 + 0.001
    rhs_minimum = torch.rand(args.k // 32, args.n, dtype=torch.float16) * 0.2 - 0.1
    lhs = pack_lhs(x, lhs_scale, lhs_sum)
    rhs = pack_rhs(q, rhs_scale, rhs_minimum)
    output = torch.empty(args.n, dtype=torch.float32)

    def run() -> None:
        _kai_q41_layout_split_kernel[(1,)](
            lhs, rhs, clamp, output, 0, args.n // 4,
            K=args.k, N=args.n, UNROLL=args.unroll,
        )

    run()
    expected = torch.zeros(args.n, dtype=torch.float32)
    for group in range(args.k // 32):
        begin = group * 32
        signed_dot = (
            x[begin : begin + 32].to(torch.int32)
            @ (q[:, begin : begin + 32].to(torch.int32) - 8).T
        )
        expected += signed_dot.float() * lhs_scale[group].float() * rhs_scale[group].float()
        expected += (rhs_minimum[group].float() + 8.0 * rhs_scale[group].float()) * lhs_sum[group].float()
    torch.testing.assert_close(output, expected, rtol=2.0e-6, atol=3.0e-5)

    latency = median_us(run, args.warmup, args.iters, args.batches)
    assembly, _, sdot, addp, stack, calls = audit_codegen(
        last_compiled(_kai_q41_layout_split_kernel), args.unroll
    )
    print(
        f"PASS KAI-style Q4_1 K={args.k} N={args.n} UNROLL={args.unroll}\n"
        f"triton_kernel_us={latency:.3f}\nasm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={sdot}\nasm_addp={addp}\n"
        f"stack_load_store={stack}\nexternal_calls={calls}\n"
        f"checksum={output.double().sum().item():.9f}"
    )


if __name__ == "__main__":
    main()
