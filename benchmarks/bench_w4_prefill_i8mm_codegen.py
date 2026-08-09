#!/usr/bin/env python3
"""Prototype ordinary-Triton Q4 x Q8 prefill through the i8mm lowering.

The benchmark deliberately keeps the quantized operands packed in memory:

* LHS: ``[M/16, K/32, 16, 32]`` signed int8 plus one FP16 scale per
  ``[M16, K32, row]``;
* RHS: ``[N/8, K/32, 16, 8]`` packed nibbles plus one FP16 scale per
  ``[K32, column]``.

Every program computes a 16x8 output tile.  A K32 group is unpacked in
registers and expressed as one ordinary ``tl.dot([16,32], [32,8])``.  The
CPU compiler must keep the K loop rolled and select SMMLA; there is no TLE or
runtime call in this experiment.
"""

from __future__ import annotations

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _w4_prefill_i8mm_kernel(
    lhs_q_ptr,
    lhs_scale_ptr,
    rhs_q4_ptr,
    rhs_scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    groups: tl.constexpr = K // 32
    k16 = tl.arange(0, 16)
    result = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for group in range(0, groups):
        packed_offset = (
            ((pid_n * groups + group) * 16 + k16[:, None]) * BLOCK_N
            + tl.arange(0, BLOCK_N)[None, :]
        )
        packed = tl.load(rhs_q4_ptr + packed_offset)
        weight_low = (packed << 4).to(tl.int8)
        weight_high = (packed & 0xF0).to(tl.int8)
        # Multiplying both nibble halves by 16 lets the epilogue use the same
        # exact fixed-point convention as KAI: SCVTF #4 after accumulation.
        weight = tl.join(weight_low, weight_high).permute(0, 2, 1).reshape(
            (32, BLOCK_N)
        )

        lhs_offset = (
            ((pid_m * groups + group) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None])
            * 32
            + tl.arange(0, 32)[None, :]
        )
        lhs = tl.load(lhs_q_ptr + lhs_offset)
        dot_scaled16 = tl.dot(lhs, weight, out_dtype=tl.int32)
        dot = dot_scaled16.to(tl.float32) * (1.0 / 16.0)

        lhs_scale = tl.load(
            lhs_scale_ptr
            + (pid_m * groups + group) * BLOCK_M
            + tl.arange(0, BLOCK_M)
        ).to(tl.float32)
        rhs_scale = tl.load(rhs_scale_ptr + group * N + cols).to(tl.float32)
        result += dot * lhs_scale[:, None] * rhs_scale[None, :]

    out_offset = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + out_offset, result)


def pack_rhs(weight: torch.Tensor, block_n: int) -> torch.Tensor:
    """Pack signed [-8, 7] weights as [N/BN, K32, 16, BN]."""
    n, k = weight.shape
    if n % block_n or k % 32:
        raise ValueError("requires N%BLOCK_N=0 and K%32=0")
    grouped = weight.reshape(n, k // 32, 32)
    low = grouped[:, :, :16].to(torch.int16) & 15
    high = grouped[:, :, 16:].to(torch.int16) & 15
    packed = (low | (high << 4)).to(torch.uint8)
    return packed.permute(0, 1, 2).reshape(n // block_n, block_n, k // 32, 16).permute(0, 2, 3, 1).contiguous()


def pack_lhs(
    lhs: torch.Tensor, scale: torch.Tensor, block_m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack K32 groups with low/high halves interleaved for SMMLA."""
    m, k = lhs.shape
    groups = k // 32
    grouped = lhs.reshape(m, groups, 32)
    interleaved = torch.stack(
        (grouped[:, :, :16], grouped[:, :, 16:]), dim=-1
    ).reshape(m, groups, 32)
    packed_lhs = (
        interleaved.reshape(m // block_m, block_m, groups, 32)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    packed_scale = (
        scale.reshape(m // block_m, block_m, groups)
        .permute(0, 2, 1)
        .contiguous()
    )
    return packed_lhs, packed_scale


def last_compiled(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    return list(cache.values())[-1]


def audit_codegen(compiled) -> tuple[int, int, int, int]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    smmla = assembly.count("smmla")
    sdot = assembly.count("sdot")
    stack = sum(
        "[sp" in line and line.lstrip().startswith(("ld", "st"))
        for line in assembly.splitlines()
    )
    calls = sum(
        " call " in line and "llvm." not in line for line in llir.splitlines()
    )
    if smmla == 0:
        raise RuntimeError("Q4 prefill did not select SMMLA")
    if "triton_cpu.dot" in llir:
        raise RuntimeError("Q4 prefill retained triton_cpu.dot")
    if calls:
        raise RuntimeError(f"Q4 prefill has {calls} external calls")
    return smmla, sdot, stack, len(assembly.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if (
        args.m % args.block_m
        or args.n % args.block_n
        or args.k % 32
        or args.block_m % 8
        or args.block_n % 4
    ):
        raise ValueError("requires tiled M, N and K32 with BM%8=0 and BN%4=0")

    torch.manual_seed(4816)
    lhs = torch.randint(-127, 128, (args.m, args.k), dtype=torch.int8)
    weight = torch.randint(-8, 8, (args.n, args.k), dtype=torch.int8)
    lhs_scale = torch.rand(args.m, args.k // 32, dtype=torch.float16) * 0.01
    rhs_scale = torch.rand(args.k // 32, args.n, dtype=torch.float16) * 0.01
    packed = pack_rhs(weight, args.block_n)
    packed_lhs, packed_lhs_scale = pack_lhs(lhs, lhs_scale, args.block_m)
    output = torch.empty((args.m, args.n), dtype=torch.float32)

    grid = (args.m // args.block_m, args.n // args.block_n)

    def run() -> None:
        _w4_prefill_i8mm_kernel[grid](
            packed_lhs,
            packed_lhs_scale,
            packed,
            rhs_scale,
            output,
            M=args.m,
            N=args.n,
            K=args.k,
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
        )

    run()
    expected = torch.zeros_like(output)
    for group in range(args.k // 32):
        begin = group * 32
        dot = lhs[:, begin : begin + 32].to(torch.int32) @ weight[:, begin : begin + 32].to(torch.int32).T
        expected += dot.float() * lhs_scale[:, group].float()[:, None] * rhs_scale[group].float()[None, :]
    torch.testing.assert_close(output, expected, rtol=3.0e-5, atol=3.0e-5)

    compiled = last_compiled(_w4_prefill_i8mm_kernel)
    smmla, sdot, stack, asm_lines = audit_codegen(compiled)
    for _ in range(args.warmup):
        run()
    begin = time.perf_counter_ns()
    for _ in range(args.iters):
        run()
    elapsed_us = (time.perf_counter_ns() - begin) / args.iters / 1000.0
    print(
        f"PASS ordinary Q4-prefill M={args.m} N={args.n} K={args.k} "
        f"BM={args.block_m} BN={args.block_n}\n"
        f"python_launch_us={elapsed_us:.3f}\n"
        f"smmla={smmla}\n"
        f"sdot={sdot}\n"
        f"stack_load_store={stack}\n"
        f"asm_lines={asm_lines}"
    )


if __name__ == "__main__":
    main()
