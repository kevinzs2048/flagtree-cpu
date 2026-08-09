#!/usr/bin/env python3
"""Ordinary-Triton W8 prefill using KleidiAI's packed M4/N4 K8 ABI."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
sys.path.insert(0, str(TRITON_PYTHON))

import torch
import triton
import triton.language as tl


def require_expected_triton() -> None:
    expected = (TRITON_PYTHON / "triton").resolve()
    actual = Path(triton.__file__).resolve()
    if expected not in actual.parents:
        raise RuntimeError(
            "wrong Triton source loaded: "
            f"expected under {expected}, got {actual}. "
            "Put TRITON_CPU_PYTHON at the front of PYTHONPATH before "
            "starting Python."
        )


require_expected_triton()


@triton.jit
def _kai_w8_prefill_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUTPUT_BF16: tl.constexpr = False,
    UNROLL_K: tl.constexpr = 1,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    panel_offsets = tl.arange(0, 128)
    lhs_panel_stride: tl.constexpr = 4 * K + 32
    rhs_panel_stride: tl.constexpr = 4 * K + 48
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

    # Both KAI operands interleave four rows in K8 units.  Load contiguous
    # physical panels, describe their ordering with ordinary join/permute,
    # and leave one logical dot visible for target-aware CPU lowering.
    for chunk in tl.range(
        0, K // 32, loop_unroll_factor=UNROLL_K
    ):
        lhs_base = pid_m * 4 * lhs_panel_stride + chunk * 128
        lhs0 = tl.load(
            lhs_packed_ptr + lhs_base + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs1 = tl.load(
            lhs_packed_ptr
            + lhs_base
            + lhs_panel_stride
            + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs2 = tl.load(
            lhs_packed_ptr
            + lhs_base
            + 2 * lhs_panel_stride
            + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs3 = tl.load(
            lhs_packed_ptr
            + lhs_base
            + 3 * lhs_panel_stride
            + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs01 = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        lhs23 = tl.join(lhs2, lhs3).permute(2, 0, 1).reshape((8, 32))
        lhs = tl.join(lhs01, lhs23).permute(2, 0, 1).reshape((16, 32))

        rhs = tl.load(
            rhs_packed_ptr
            + pid_n * rhs_panel_stride
            + chunk * 128
            + panel_offsets
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.int32)

    # Metadata is physically four contiguous values at the end of each M4
    # panel.  Preserve those vector loads in the Triton program instead of
    # constructing sixteen unrelated pointers and relying on gather folding.
    meta_lanes = tl.arange(0, 4)
    lhs_meta = pid_m * 4 * lhs_panel_stride + 4 * K
    lhs_offset0 = tl.load(
        (lhs_packed_ptr + lhs_meta).to(tl.pointer_type(tl.int32))
        + meta_lanes
    )
    lhs_offset1 = tl.load(
        (lhs_packed_ptr + lhs_meta + lhs_panel_stride).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    lhs_offset2 = tl.load(
        (lhs_packed_ptr + lhs_meta + 2 * lhs_panel_stride).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    lhs_offset3 = tl.load(
        (lhs_packed_ptr + lhs_meta + 3 * lhs_panel_stride).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    lhs_offset01 = tl.join(lhs_offset0, lhs_offset1).permute(1, 0).reshape(8)
    lhs_offset23 = tl.join(lhs_offset2, lhs_offset3).permute(1, 0).reshape(8)
    lhs_offset = tl.join(lhs_offset01, lhs_offset23).permute(1, 0).reshape(16)

    lhs_scale0 = tl.load(
        (lhs_packed_ptr + lhs_meta + 16).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    lhs_scale1 = tl.load(
        (lhs_packed_ptr + lhs_meta + lhs_panel_stride + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    lhs_scale2 = tl.load(
        (lhs_packed_ptr + lhs_meta + 2 * lhs_panel_stride + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    lhs_scale3 = tl.load(
        (lhs_packed_ptr + lhs_meta + 3 * lhs_panel_stride + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    lhs_scale01 = tl.join(lhs_scale0, lhs_scale1).permute(1, 0).reshape(8)
    lhs_scale23 = tl.join(lhs_scale2, lhs_scale3).permute(1, 0).reshape(8)
    lhs_scale = tl.join(lhs_scale01, lhs_scale23).permute(1, 0).reshape(16)

    rhs_meta = pid_n * rhs_panel_stride + 4 * K
    rhs_sum_ptr = (rhs_packed_ptr + rhs_meta).to(tl.pointer_type(tl.int32))
    rhs_scale_ptr = (rhs_packed_ptr + rhs_meta + 16).to(
        tl.pointer_type(tl.float32)
    )
    bias_ptr = (rhs_packed_ptr + rhs_meta + 32).to(
        tl.pointer_type(tl.float32)
    )
    rhs_sum = tl.load(rhs_sum_ptr + cols)
    rhs_scale = tl.load(rhs_scale_ptr + cols)
    bias = tl.load(bias_ptr + cols)
    corrected = accumulator + lhs_offset[:, None] * rhs_sum[None, :]
    # KAI forms the pairwise scale product before multiplying the converted
    # accumulator.  Preserve that association so final BF16 ties agree with
    # the reference microkernel instead of depending on reassociation.
    combined_scale = lhs_scale[:, None] * rhs_scale[None, :]
    result = (
        corrected.to(tl.float32) * combined_scale + bias[None, :]
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)
    result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
    if OUTPUT_BF16:
        result = result.to(tl.bfloat16)
    output_rows = pid_m * BLOCK_M + rows
    output_cols = pid_n * BLOCK_N + cols
    tl.store(
        out_ptr + output_rows[:, None] * N + output_cols[None, :], result
    )


@triton.jit
def _kai_w8_prefill_short_tail_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUTPUT_BF16: tl.constexpr = False,
    UNROLL_K: tl.constexpr = 1,
):
    """Exact-KAI asymmetric M4/M8 prefill without padded M16 work."""
    pid_n = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    panel_offsets = tl.arange(0, 128)
    meta_lanes = tl.arange(0, 4)
    lhs_panel_stride: tl.constexpr = 4 * K + 32
    rhs_panel_stride: tl.constexpr = 4 * K + 48
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

    for chunk in tl.range(
        0, K // 32, loop_unroll_factor=UNROLL_K
    ):
        lhs_base = chunk * 128
        rhs_base = pid_n * rhs_panel_stride + chunk * 128
        rhs = tl.load(
            rhs_packed_ptr + rhs_base + panel_offsets
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        lhs0 = tl.load(
            lhs_packed_ptr + lhs_base + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        if BLOCK_M == 4:
            lhs = lhs0
        else:
            lhs1 = tl.load(
                lhs_packed_ptr
                + lhs_base
                + lhs_panel_stride
                + panel_offsets
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.int32)

    rhs_meta = pid_n * rhs_panel_stride + 4 * K
    rhs_sum = tl.load(
        (rhs_packed_ptr + rhs_meta).to(tl.pointer_type(tl.int32)) + cols
    )
    rhs_scale = tl.load(
        (rhs_packed_ptr + rhs_meta + 16).to(tl.pointer_type(tl.float32))
        + cols
    )
    bias = tl.load(
        (rhs_packed_ptr + rhs_meta + 32).to(tl.pointer_type(tl.float32))
        + cols
    )
    lhs_offset0 = tl.load(
        (lhs_packed_ptr + 4 * K).to(tl.pointer_type(tl.int32))
        + meta_lanes
    )
    lhs_scale0 = tl.load(
        (lhs_packed_ptr + 4 * K + 16).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    if BLOCK_M == 4:
        lhs_offset = lhs_offset0
        lhs_scale = lhs_scale0
    else:
        lhs_offset1 = tl.load(
            (lhs_packed_ptr + lhs_panel_stride + 4 * K).to(
                tl.pointer_type(tl.int32)
            )
            + meta_lanes
        )
        lhs_scale1 = tl.load(
            (lhs_packed_ptr + lhs_panel_stride + 4 * K + 16).to(
                tl.pointer_type(tl.float32)
            )
            + meta_lanes
        )
        lhs_offset = tl.join(lhs_offset0, lhs_offset1).permute(
            1, 0
        ).reshape((8,))
        lhs_scale = tl.join(lhs_scale0, lhs_scale1).permute(
            1, 0
        ).reshape((8,))
    corrected = accumulator + lhs_offset[:, None] * rhs_sum[None, :]
    combined_scale = lhs_scale[:, None] * rhs_scale[None, :]
    result = corrected.to(tl.float32) * combined_scale + bias[None, :]
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)
    result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
    if OUTPUT_BF16:
        result = result.to(tl.bfloat16)
    rows = tl.arange(0, BLOCK_M)
    output_cols = pid_n * BLOCK_N + cols
    tl.store(
        out_ptr + rows[:, None] * N + output_cols[None, :], result
    )


@triton.jit
def _kai_w8_prefill_m12_tail_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUTPUT_BF16: tl.constexpr = False,
    UNROLL_K: tl.constexpr = 1,
):
    """Exact-KAI asymmetric M12 as compiler-fused M8 plus M4 dots."""
    pid_n = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    panel_offsets = tl.arange(0, 128)
    meta_lanes = tl.arange(0, 4)
    lhs_panel_stride: tl.constexpr = 4 * K + 32
    rhs_panel_stride: tl.constexpr = 4 * K + 48
    accumulator8 = tl.zeros((8, BLOCK_N), tl.int32)
    accumulator4 = tl.zeros((4, BLOCK_N), tl.int32)

    for chunk in tl.range(
        0, K // 32, loop_unroll_factor=UNROLL_K
    ):
        lhs_base = chunk * 128
        rhs_base = pid_n * rhs_panel_stride + chunk * 128
        rhs = tl.load(
            rhs_packed_ptr + rhs_base + panel_offsets
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        lhs0 = tl.load(
            lhs_packed_ptr + lhs_base + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs1 = tl.load(
            lhs_packed_ptr
            + lhs_base
            + lhs_panel_stride
            + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs2 = tl.load(
            lhs_packed_ptr
            + lhs_base
            + 2 * lhs_panel_stride
            + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs8 = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        accumulator8 += tl.dot(lhs8, rhs, out_dtype=tl.int32)
        accumulator4 += tl.dot(lhs2, rhs, out_dtype=tl.int32)

    rhs_meta = pid_n * rhs_panel_stride + 4 * K
    rhs_sum = tl.load(
        (rhs_packed_ptr + rhs_meta).to(tl.pointer_type(tl.int32)) + cols
    )
    rhs_scale = tl.load(
        (rhs_packed_ptr + rhs_meta + 16).to(tl.pointer_type(tl.float32))
        + cols
    )
    bias = tl.load(
        (rhs_packed_ptr + rhs_meta + 32).to(tl.pointer_type(tl.float32))
        + cols
    )
    offset0 = tl.load(
        (lhs_packed_ptr + 4 * K).to(tl.pointer_type(tl.int32))
        + meta_lanes
    )
    offset1 = tl.load(
        (lhs_packed_ptr + lhs_panel_stride + 4 * K).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    offset2 = tl.load(
        (lhs_packed_ptr + 2 * lhs_panel_stride + 4 * K).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    scale0 = tl.load(
        (lhs_packed_ptr + 4 * K + 16).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    scale1 = tl.load(
        (lhs_packed_ptr + lhs_panel_stride + 4 * K + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    scale2 = tl.load(
        (lhs_packed_ptr + 2 * lhs_panel_stride + 4 * K + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    offset8 = tl.join(offset0, offset1).permute(1, 0).reshape((8,))
    scale8 = tl.join(scale0, scale1).permute(1, 0).reshape((8,))
    corrected8 = accumulator8 + offset8[:, None] * rhs_sum[None, :]
    corrected4 = accumulator4 + offset2[:, None] * rhs_sum[None, :]
    result8 = (
        corrected8.to(tl.float32)
        * (scale8[:, None] * rhs_scale[None, :])
        + bias[None, :]
    )
    result4 = (
        corrected4.to(tl.float32)
        * (scale2[:, None] * rhs_scale[None, :])
        + bias[None, :]
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)
    result8 = tl.minimum(tl.maximum(result8, clamp_min), clamp_max)
    result4 = tl.minimum(tl.maximum(result4, clamp_min), clamp_max)
    if OUTPUT_BF16:
        result8 = result8.to(tl.bfloat16)
        result4 = result4.to(tl.bfloat16)
    output_cols = pid_n * BLOCK_N + cols
    tl.store(
        out_ptr + tl.arange(0, 8)[:, None] * N + output_cols[None, :],
        result8,
    )
    tl.store(
        out_ptr + (8 + meta_lanes)[:, None] * N + output_cols[None, :],
        result4,
    )


def pack_lhs(
    values: torch.Tensor, offsets: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    m, k = values.shape
    if m % 4 or k % 32:
        raise ValueError("KAI W8 prefill LHS requires M%4=0 and K%32=0")
    stride = 4 * k + 32
    packed = torch.zeros((m // 4, stride), dtype=torch.int8)
    interleaved = (
        values.reshape(m // 4, 4, k // 8, 8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m // 4, 4 * k)
    )
    packed[:, : 4 * k].copy_(interleaved)
    packed[:, 4 * k : 4 * k + 16].view(torch.int32).copy_(
        offsets.reshape(m // 4, 4)
    )
    packed[:, 4 * k + 16 : 4 * k + 32].view(torch.float32).copy_(
        scales.reshape(m // 4, 4)
    )
    return packed.reshape(-1)


def pack_rhs(
    values: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = values.shape
    if n % 4 or k % 32:
        raise ValueError("KAI W8 prefill RHS requires N%4=0 and K%32=0")
    stride = 4 * k + 48
    packed = torch.zeros((n // 4, stride), dtype=torch.int8)
    interleaved = (
        values.reshape(n // 4, 4, k // 8, 8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(n // 4, 4 * k)
    )
    packed[:, : 4 * k].copy_(interleaved)
    sums = values.to(torch.int32).sum(dim=1)
    packed[:, 4 * k : 4 * k + 16].view(torch.int32).copy_(
        sums.reshape(n // 4, 4)
    )
    packed[:, 4 * k + 16 : 4 * k + 32].view(torch.float32).copy_(
        scales.reshape(n // 4, 4)
    )
    packed[:, 4 * k + 32 : 4 * k + 48].view(torch.float32).copy_(
        bias.reshape(n // 4, 4)
    )
    return packed.reshape(-1), sums


def last_compiled(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    return list(cache.values())[-1]


def bf16_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual_bits = actual.view(torch.int16).to(torch.int32) & 0xFFFF
    expected_bits = expected.view(torch.int16).to(torch.int32) & 0xFFFF

    def ordered(bits: torch.Tensor) -> torch.Tensor:
        magnitude = bits & 0x7FFF
        return torch.where(
            (bits & 0x8000) != 0,
            0x8000 - magnitude,
            0x8000 + magnitude,
        )

    return (ordered(actual_bits) - ordered(expected_bits)).abs()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-bf16", action="store_true")
    parser.add_argument("--unroll-k", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.m not in (4, 8, 12, 16) or args.n % 4 or args.k % 32:
        raise ValueError("requires M in {4,8,12,16}, N%4=0 and K%32=0")
    if (args.k // 32) % args.unroll_k:
        raise ValueError("K/32 must be divisible by the K-loop unroll factor")

    torch.manual_seed(8841)
    lhs = torch.randint(-127, 128, (args.m, args.k), dtype=torch.int8)
    rhs = torch.randint(-127, 128, (args.n, args.k), dtype=torch.int8)
    lhs_offset = torch.randint(-24, 25, (args.m,), dtype=torch.int32)
    lhs_scale = torch.rand(args.m, dtype=torch.float32) * 0.01
    rhs_scale = torch.rand(args.n, dtype=torch.float32) * 0.01
    bias = torch.rand(args.n, dtype=torch.float32) - 0.5
    lhs_packed = pack_lhs(lhs, lhs_offset, lhs_scale)
    rhs_packed, rhs_sum = pack_rhs(rhs, rhs_scale, bias)
    clamp = torch.tensor([-7.0, 6.0], dtype=torch.float32)
    output = torch.empty(
        (args.m, args.n),
        dtype=torch.bfloat16 if args.output_bf16 else torch.float32,
    )
    grid = (1, args.n // 4)
    if args.m <= 8:
        selected_kernel = _kai_w8_prefill_short_tail_kernel
    elif args.m == 12:
        selected_kernel = _kai_w8_prefill_m12_tail_kernel
    else:
        selected_kernel = _kai_w8_prefill_kernel

    def run() -> None:
        if args.m == 12:
            selected_kernel[grid](
                lhs_packed,
                rhs_packed,
                clamp,
                output,
                M=args.m,
                N=args.n,
                K=args.k,
                BLOCK_N=4,
                OUTPUT_BF16=args.output_bf16,
                UNROLL_K=args.unroll_k,
            )
        else:
            selected_kernel[grid](
                lhs_packed,
                rhs_packed,
                clamp,
                output,
                M=args.m,
                N=args.n,
                K=args.k,
                BLOCK_M=args.m,
                BLOCK_N=4,
                OUTPUT_BF16=args.output_bf16,
                UNROLL_K=args.unroll_k,
            )

    run()
    dot = lhs.to(torch.int32) @ rhs.to(torch.int32).T
    corrected = dot + lhs_offset[:, None] * rhs_sum[None, :]
    combined_scale = lhs_scale[:, None] * rhs_scale[None, :]
    # The generated epilogue and the official KAI kernel both contract the
    # final multiply/add to FMLA.  A standalone PyTorch multiply followed by
    # add rounds the FP32 product first and can differ by two BF16 ULP close
    # to zero.  addcmul has the required fused semantics on Arm CPU.
    expected = torch.addcmul(
        bias[None, :], corrected.float(), combined_scale
    ).clamp(clamp[0], clamp[1])
    python_max_bf16_ulp = 0
    if args.output_bf16:
        expected_bf16 = expected.to(torch.bfloat16)
        python_max_bf16_ulp = int(
            bf16_ulp_distance(output, expected_bf16).max()
        )
        if python_max_bf16_ulp != 0:
            raise AssertionError(
                "BF16 output differs from the fused PyTorch expression by "
                f"{python_max_bf16_ulp} ULP"
            )
    else:
        torch.testing.assert_close(
            output, expected, rtol=3.0e-5, atol=3.0e-5
        )

    compiled = last_compiled(selected_kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    smmla = assembly.count("smmla")
    stack = sum(
        "[sp" in line and line.lstrip().startswith(("ld", "st"))
        for line in assembly.splitlines()
    )
    calls = sum(
        " call " in line and "llvm." not in line
        for line in llir.splitlines()
    )
    expected_smmla = 4 * args.m * args.unroll_k
    if smmla != expected_smmla or "triton_cpu.dot" in llir or calls:
        raise RuntimeError(
            f"bad codegen: SMMLA={smmla}, residual_dot="
            f"{'triton_cpu.dot' in llir}, external_calls={calls}"
        )
    for _ in range(args.warmup):
        run()
    start = time.perf_counter_ns()
    for _ in range(args.iters):
        run()
    elapsed_us = (time.perf_counter_ns() - start) / args.iters / 1000.0
    print(
        f"PASS KAI-layout W8-prefill M={args.m} N={args.n} K={args.k}\n"
        f"python_launch_us={elapsed_us:.3f}\n"
        f"smmla={smmla}\n"
        f"stack_load_store={stack}\n"
        f"external_calls={calls}\n"
        f"python_max_bf16_ulp={python_max_bf16_ulp}\n"
        f"asm_lines={len(assembly.splitlines())}"
    )


if __name__ == "__main__":
    main()
