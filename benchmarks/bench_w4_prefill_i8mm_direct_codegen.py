#!/usr/bin/env python3
"""Ordinary-Triton Q4 x Q8 prefill with an SMMLA-oriented RHS layout.

The packed RHS layout is ``[N/BN, K/32, 4, BN, 4]``.  The Triton source
reconstructs a logical 32xBN operand and still uses one ordinary ``tl.dot``.
The CPU compiler can recognize the layout-preserving reshape/permute chain
and load each eight-byte, two-column panel directly from packed memory.
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
    result = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for group in range(0, groups):
        packed_base = (pid_n * groups + group) * 16 * BLOCK_N
        packed_flat = tl.load(
            rhs_q4_ptr
            + packed_base
            + tl.arange(0, 16 * BLOCK_N)
        )
        packed_direct = packed_flat.reshape((4, BLOCK_N, 4))
        packed = packed_direct.permute(0, 2, 1).reshape(
            (16, BLOCK_N)
        )
        weight_low = (packed << 4).to(tl.int8)
        weight_high = (packed & 0xF0).to(tl.int8)
        weight = tl.join(weight_low, weight_high).permute(0, 2, 1).reshape(
            (32, BLOCK_N)
        )

        lhs_offset = (
            (((pid_m * groups + group) * BLOCK_M
              + tl.arange(0, BLOCK_M)[:, None]) * 32)
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
        rhs_scale = tl.load(
            rhs_scale_ptr + group * N + cols
        ).to(tl.float32)
        result += dot * lhs_scale[:, None] * rhs_scale[None, :]

    out_offset = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + out_offset, result)


@triton.jit
def _w4_prefill_i8mm_m4_kernel(
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
    """KAI-style 16x4 tile evaluated as four sequential M4 subtiles."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    groups: tl.constexpr = K // 32
    lanes_m4 = tl.arange(0, 4)
    result0 = tl.zeros((4, BLOCK_N), tl.float32)
    result1 = tl.zeros((4, BLOCK_N), tl.float32)
    result2 = tl.zeros((4, BLOCK_N), tl.float32)
    result3 = tl.zeros((4, BLOCK_N), tl.float32)

    for group in range(0, groups):
        packed_base = (pid_n * groups + group) * 16 * BLOCK_N
        packed_flat = tl.load(
            rhs_q4_ptr + packed_base + tl.arange(0, 16 * BLOCK_N)
        )
        packed_direct = packed_flat.reshape((4, BLOCK_N, 4))
        packed = packed_direct.permute(0, 2, 1).reshape((16, BLOCK_N))
        weight_low = (packed << 4).to(tl.int8)
        weight_high = (packed & 0xF0).to(tl.int8)
        weight = tl.join(weight_low, weight_high).permute(0, 2, 1).reshape(
            (32, BLOCK_N)
        )

        lhs_group_base = (pid_m * 4 * groups + group) * 128
        lhs_panel_stride = groups * 128
        lhs_scale_base = (pid_m * groups + group) * BLOCK_M
        rhs_scale_group = tl.load(
            rhs_scale_ptr + group * N + cols
        ).to(tl.float32)

        lhs0 = tl.load(
            lhs_q_ptr + lhs_group_base + tl.arange(0, 128)
        ).reshape((4, 2, 2, 8)).permute(1, 2, 0, 3).reshape((4, 32))
        lhs_scale0 = tl.load(
            lhs_scale_ptr + lhs_scale_base + lanes_m4
        ).to(tl.float32)
        dot0 = tl.dot(lhs0, weight, out_dtype=tl.int32)
        result0 += (
            dot0.to(tl.float32)
            * (1.0 / 16.0)
            * lhs_scale0[:, None]
            * rhs_scale_group[None, :]
        )

        lhs1 = tl.load(
            lhs_q_ptr
            + lhs_group_base
            + lhs_panel_stride
            + tl.arange(0, 128)
        ).reshape((4, 2, 2, 8)).permute(1, 2, 0, 3).reshape((4, 32))
        lhs_scale1 = tl.load(
            lhs_scale_ptr + lhs_scale_base + 4 + lanes_m4
        ).to(tl.float32)
        dot1 = tl.dot(lhs1, weight, out_dtype=tl.int32)
        result1 += (
            dot1.to(tl.float32)
            * (1.0 / 16.0)
            * lhs_scale1[:, None]
            * rhs_scale_group[None, :]
        )

        lhs2 = tl.load(
            lhs_q_ptr
            + lhs_group_base
            + 2 * lhs_panel_stride
            + tl.arange(0, 128)
        ).reshape((4, 2, 2, 8)).permute(1, 2, 0, 3).reshape((4, 32))
        lhs_scale2 = tl.load(
            lhs_scale_ptr + lhs_scale_base + 8 + lanes_m4
        ).to(tl.float32)
        dot2 = tl.dot(lhs2, weight, out_dtype=tl.int32)
        result2 += (
            dot2.to(tl.float32)
            * (1.0 / 16.0)
            * lhs_scale2[:, None]
            * rhs_scale_group[None, :]
        )

        lhs3 = tl.load(
            lhs_q_ptr
            + lhs_group_base
            + 3 * lhs_panel_stride
            + tl.arange(0, 128)
        ).reshape((4, 2, 2, 8)).permute(1, 2, 0, 3).reshape((4, 32))
        lhs_scale3 = tl.load(
            lhs_scale_ptr + lhs_scale_base + 12 + lanes_m4
        ).to(tl.float32)
        dot3 = tl.dot(lhs3, weight, out_dtype=tl.int32)
        result3 += (
            dot3.to(tl.float32)
            * (1.0 / 16.0)
            * lhs_scale3[:, None]
            * rhs_scale_group[None, :]
        )

    output_base_row = pid_m * BLOCK_M
    tl.store(
        out_ptr
        + (output_base_row + lanes_m4)[:, None] * N
        + cols[None, :],
        result0,
    )
    tl.store(
        out_ptr
        + (output_base_row + 4 + lanes_m4)[:, None] * N
        + cols[None, :],
        result1,
    )
    tl.store(
        out_ptr
        + (output_base_row + 8 + lanes_m4)[:, None] * N
        + cols[None, :],
        result2,
    )
    tl.store(
        out_ptr
        + (output_base_row + 12 + lanes_m4)[:, None] * N
        + cols[None, :],
        result3,
    )


@triton.jit
def _w4_prefill_i8mm_m4_kai_kernel(
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
    PACKED_ABI: tl.constexpr,
):
    """M4-M16 x N4 tile using KAI's native i8mm physical panels."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    groups: tl.constexpr = K // 32
    num_panels: tl.constexpr = BLOCK_M // 4
    lanes_m4 = tl.arange(0, 4)
    result0 = tl.zeros((4, BLOCK_N), tl.float32)
    if BLOCK_M >= 8:
        result1 = tl.zeros((4, BLOCK_N), tl.float32)
    if BLOCK_M >= 12:
        result2 = tl.zeros((4, BLOCK_N), tl.float32)
    if BLOCK_M >= 16:
        result3 = tl.zeros((4, BLOCK_N), tl.float32)

    for group in range(0, groups):
        # KAI stores two K8 segments.  Every column contributes eight bytes;
        # the low and high nibbles represent K[0:16] and K[16:32].
        if PACKED_ABI:
            packed_base = (pid_n * groups + group) * 72 + 8
        else:
            packed_base = (pid_n * groups + group) * 16 * BLOCK_N
        packed_flat = tl.load(
            rhs_q4_ptr + packed_base + tl.arange(0, 16 * BLOCK_N)
        )
        packed_direct = packed_flat.reshape((2, BLOCK_N, 8))
        packed = packed_direct.permute(0, 2, 1).reshape((16, BLOCK_N))
        weight_low = (packed << 4).to(tl.int8)
        weight_high = (packed & 0xF0).to(tl.int8)
        weight = tl.join(weight_low, weight_high).permute(
            0, 2, 1
        ).reshape((32, BLOCK_N))

        if PACKED_ABI:
            lhs_group_base = (
                pid_m * num_panels * groups + group
            ) * 136 + 8
            lhs_panel_stride = groups * 136
            lhs_scale_base = (
                pid_m * num_panels * groups * 68 + group * 68
            )
            lhs_scale_panel_stride = groups * 68
            rhs_scale_base = (pid_n * groups + group) * 36
            rhs_scale_group = tl.load(
                rhs_scale_ptr + rhs_scale_base + tl.arange(0, BLOCK_N)
            ).to(tl.float32)
        else:
            lhs_group_base = (
                pid_m * num_panels * groups + group
            ) * 128
            lhs_panel_stride = groups * 128
            lhs_scale_base = (pid_m * groups + group) * BLOCK_M
            lhs_scale_panel_stride = 4
            rhs_scale_group = tl.load(
                rhs_scale_ptr + group * N + cols
            ).to(tl.float32)

        lhs0_sequential = tl.load(
            lhs_q_ptr + lhs_group_base + tl.arange(0, 128)
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs0 = lhs0_sequential.reshape((4, 2, 16)).permute(
            0, 2, 1
        ).reshape((4, 32))
        lhs_scale0 = tl.load(
            lhs_scale_ptr + lhs_scale_base + lanes_m4
        ).to(tl.float32)
        dot0 = tl.dot(lhs0, weight, out_dtype=tl.int32)
        result0 += (
            dot0.to(tl.float32)
            * (1.0 / 16.0)
            * lhs_scale0[:, None]
            * rhs_scale_group[None, :]
        )

        if BLOCK_M >= 8:
            lhs1_sequential = tl.load(
                lhs_q_ptr
                + lhs_group_base
                + lhs_panel_stride
                + tl.arange(0, 128)
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs1 = lhs1_sequential.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            lhs_scale1 = tl.load(
                lhs_scale_ptr
                + lhs_scale_base
                + lhs_scale_panel_stride
                + lanes_m4
            ).to(tl.float32)
            dot1 = tl.dot(lhs1, weight, out_dtype=tl.int32)
            result1 += (
                dot1.to(tl.float32)
                * (1.0 / 16.0)
                * lhs_scale1[:, None]
                * rhs_scale_group[None, :]
            )

        if BLOCK_M >= 12:
            lhs2_sequential = tl.load(
                lhs_q_ptr
                + lhs_group_base
                + 2 * lhs_panel_stride
                + tl.arange(0, 128)
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs2 = lhs2_sequential.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            lhs_scale2 = tl.load(
                lhs_scale_ptr
                + lhs_scale_base
                + 2 * lhs_scale_panel_stride
                + lanes_m4
            ).to(tl.float32)
            dot2 = tl.dot(lhs2, weight, out_dtype=tl.int32)
            result2 += (
                dot2.to(tl.float32)
                * (1.0 / 16.0)
                * lhs_scale2[:, None]
                * rhs_scale_group[None, :]
            )

        if BLOCK_M >= 16:
            lhs3_sequential = tl.load(
                lhs_q_ptr
                + lhs_group_base
                + 3 * lhs_panel_stride
                + tl.arange(0, 128)
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs3 = lhs3_sequential.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            lhs_scale3 = tl.load(
                lhs_scale_ptr
                + lhs_scale_base
                + 3 * lhs_scale_panel_stride
                + lanes_m4
            ).to(tl.float32)
            dot3 = tl.dot(lhs3, weight, out_dtype=tl.int32)
            result3 += (
                dot3.to(tl.float32)
                * (1.0 / 16.0)
                * lhs_scale3[:, None]
                * rhs_scale_group[None, :]
            )

    output_base_row = pid_m * BLOCK_M
    tl.store(
        out_ptr
        + (output_base_row + lanes_m4)[:, None] * N
        + cols[None, :],
        result0,
    )
    if BLOCK_M >= 8:
        tl.store(
            out_ptr
            + (output_base_row + 4 + lanes_m4)[:, None] * N
            + cols[None, :],
            result1,
        )
    if BLOCK_M >= 12:
        tl.store(
            out_ptr
            + (output_base_row + 8 + lanes_m4)[:, None] * N
            + cols[None, :],
            result2,
        )
    if BLOCK_M >= 16:
        tl.store(
            out_ptr
            + (output_base_row + 12 + lanes_m4)[:, None] * N
            + cols[None, :],
            result3,
        )


@triton.jit
def _w4_prefill_i8mm_serial_n_kernel(
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
    """Roll output-column tiles inside one program to remove grid calls."""
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    groups: tl.constexpr = K // 32

    for tile_n in range(0, N // BLOCK_N):
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        result = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for group in range(0, groups):
            packed_base = (tile_n * groups + group) * 16 * BLOCK_N
            packed_flat = tl.load(
                rhs_q4_ptr
                + packed_base
                + tl.arange(0, 16 * BLOCK_N)
            )
            packed_direct = packed_flat.reshape((4, BLOCK_N, 4))
            packed = packed_direct.permute(0, 2, 1).reshape(
                (16, BLOCK_N)
            )
            weight_low = (packed << 4).to(tl.int8)
            weight_high = (packed & 0xF0).to(tl.int8)
            weight = tl.join(weight_low, weight_high).permute(
                0, 2, 1
            ).reshape((32, BLOCK_N))

            lhs_offset = (
                (((pid_m * groups + group) * BLOCK_M
                  + tl.arange(0, BLOCK_M)[:, None]) * 32)
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
            rhs_scale = tl.load(
                rhs_scale_ptr + group * N + cols
            ).to(tl.float32)
            result += dot * lhs_scale[:, None] * rhs_scale[None, :]

        out_offset = rows[:, None] * N + cols[None, :]
        tl.store(out_ptr + out_offset, result)


def pack_rhs(weight: torch.Tensor, block_n: int) -> torch.Tensor:
    """Pack signed [-8, 7] weights as [N/BN, K32, K4, BN, 4]."""
    n, k = weight.shape
    if n % block_n or k % 32:
        raise ValueError("requires N%BLOCK_N=0 and K%32=0")
    grouped = weight.reshape(n, k // 32, 32)
    low = grouped[:, :, :16].to(torch.int16) & 15
    high = grouped[:, :, 16:].to(torch.int16) & 15
    packed = (low | (high << 4)).to(torch.uint8)
    return (
        packed.reshape(n // block_n, block_n, k // 32, 4, 4)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )


def pack_rhs_kai(weight: torch.Tensor, block_n: int) -> torch.Tensor:
    """Pack KAI-style [N/BN, K32, K8-segment, BN, packed-K8]."""
    n, k = weight.shape
    if block_n != 4 or n % block_n or k % 32:
        raise ValueError("KAI packing requires BLOCK_N=4 and K%32=0")
    grouped = weight.reshape(n, k // 32, 32)
    low = grouped[:, :, :16].reshape(n, k // 32, 2, 8)
    high = grouped[:, :, 16:].reshape(n, k // 32, 2, 8)
    packed = ((low.to(torch.int16) & 15) |
              ((high.to(torch.int16) & 15) << 4)).to(torch.uint8)
    return (
        packed.reshape(n // block_n, block_n, k // 32, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )


def pack_lhs(
    lhs: torch.Tensor, scale: torch.Tensor, block_m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack each K32 group as k0,k16,k1,k17,... for nibble expansion."""
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


def pack_lhs_m4(
    lhs: torch.Tensor, scale: torch.Tensor, block_m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack K8 x row-pair panels consumed directly by NEON i8mm."""
    m, k = lhs.shape
    groups = k // 32
    grouped = lhs.reshape(m, groups, 32)
    interleaved = torch.stack(
        (grouped[:, :, :16], grouped[:, :, 16:]), dim=-1
    ).reshape(m, groups, 32)
    packed_lhs = (
        interleaved.reshape(m // 4, 2, 2, groups, 4, 8)
        .permute(0, 3, 4, 1, 2, 5)
        .contiguous()
    )
    packed_scale = (
        scale.reshape(m // block_m, block_m, groups)
        .permute(0, 2, 1)
        .contiguous()
    )
    return packed_lhs, packed_scale


def pack_lhs_m4_kai(
    lhs: torch.Tensor, scale: torch.Tensor, block_m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack four rows as KAI's [K8, row, 8] i8mm panels."""
    m, k = lhs.shape
    groups = k // 32
    packed_lhs = (
        lhs.reshape(m // 4, 4, groups, 4, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    packed_scale = (
        scale.reshape(m // block_m, block_m, groups)
        .permute(0, 2, 1)
        .contiguous()
    )
    return packed_lhs, packed_scale


def pack_kai_abi(
    lhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    weight: torch.Tensor,
    rhs_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact scale+panel blobs consumed by KAI's 16x4 kernel."""
    m, k = lhs.shape
    n = weight.shape[0]
    groups = k // 32
    lhs_data, _ = pack_lhs_m4_kai(lhs, lhs_scale, 4)
    lhs_data = lhs_data.reshape(m // 4, groups, 128).view(torch.uint8)
    lhs_scales = (
        lhs_scale.reshape(m // 4, 4, groups)
        .permute(0, 2, 1)
        .contiguous()
        .view(torch.uint8)
        .reshape(m // 4, groups, 8)
    )
    lhs_blob = torch.empty((m // 4, groups, 136), dtype=torch.uint8)
    lhs_blob[:, :, :8] = lhs_scales
    lhs_blob[:, :, 8:] = lhs_data

    rhs_data = pack_rhs_kai(weight, 4).reshape(n // 4, groups, 64)
    rhs_scales = (
        rhs_scale.T.reshape(n // 4, 4, groups)
        .permute(0, 2, 1)
        .contiguous()
        .view(torch.uint8)
        .reshape(n // 4, groups, 8)
    )
    rhs_blob = torch.empty((n // 4, groups, 72), dtype=torch.uint8)
    rhs_blob[:, :, :8] = rhs_scales
    rhs_blob[:, :, 8:] = rhs_data
    return lhs_blob, rhs_blob


def last_compiled(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    return list(cache.values())[-1]


def audit_codegen(compiled) -> tuple[int, int, int, int, int]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    smmla = assembly.count("smmla")
    sdot = assembly.count("sdot")
    zip1 = sum(
        line.lstrip().startswith("zip1 ")
        for line in assembly.splitlines()
    )
    stack = sum(
        "[sp" in line and line.lstrip().startswith(("ld", "st"))
        for line in assembly.splitlines()
    )
    calls = sum(
        " call " in line and "llvm." not in line
        for line in llir.splitlines()
    )
    if smmla == 0 or "triton_cpu.dot" in llir or calls:
        raise RuntimeError(
            f"bad codegen: SMMLA={smmla}, residual_dot="
            f"{'triton_cpu.dot' in llir}, external_calls={calls}"
        )
    return smmla, sdot, zip1, stack, len(assembly.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--serial-n", action="store_true")
    parser.add_argument("--m4-schedule", action="store_true")
    parser.add_argument("--kai-layout", action="store_true")
    parser.add_argument("--kai-abi", action="store_true")
    args = parser.parse_args()
    if (
        args.m % args.block_m
        or args.n % args.block_n
        or args.k % 32
        or (not args.kai_layout and args.block_m % 8)
        or args.block_n not in (4, 8)
        or (
            args.m4_schedule
            and (
                args.block_n != 4
                or (
                    args.kai_layout
                    and args.block_m not in (4, 8, 12, 16)
                )
                or (not args.kai_layout and args.block_m != 16)
            )
        )
        or (args.kai_layout and not args.m4_schedule)
        or (args.kai_abi and not args.kai_layout)
    ):
        raise ValueError(
            "requires tiled M, N and K32 with BN=4/8; generic BM%8=0; "
            "KAI M4 schedule requires BM in {4,8,12,16}, BN=4"
        )

    torch.manual_seed(4817)
    lhs = torch.randint(-127, 128, (args.m, args.k), dtype=torch.int8)
    weight = torch.randint(-8, 8, (args.n, args.k), dtype=torch.int8)
    lhs_scale = (
        torch.rand(args.m, args.k // 32, dtype=torch.float16) * 0.01
    )
    rhs_scale = (
        torch.rand(args.k // 32, args.n, dtype=torch.float16) * 0.01
    )
    if args.kai_abi:
        packed_lhs, packed = pack_kai_abi(
            lhs, lhs_scale, weight, rhs_scale
        )
        packed_lhs_q = packed_lhs.view(torch.int8)
        packed_lhs_scale = packed_lhs.view(torch.float16)
        packed_rhs_scale = packed.view(torch.float16)
    else:
        packed = (
            pack_rhs_kai(weight, args.block_n)
            if args.kai_layout
            else pack_rhs(weight, args.block_n)
        )
        if args.kai_layout:
            pack_lhs_fn = pack_lhs_m4_kai
        else:
            pack_lhs_fn = pack_lhs_m4 if args.m4_schedule else pack_lhs
        packed_lhs, packed_lhs_scale = pack_lhs_fn(
            lhs, lhs_scale, args.block_m
        )
        packed_lhs_q = packed_lhs
        packed_rhs_scale = rhs_scale
    output = torch.empty((args.m, args.n), dtype=torch.float32)
    if args.kai_layout:
        kernel = _w4_prefill_i8mm_m4_kai_kernel
    elif args.m4_schedule:
        kernel = _w4_prefill_i8mm_m4_kernel
    elif args.serial_n:
        kernel = _w4_prefill_i8mm_serial_n_kernel
    else:
        kernel = _w4_prefill_i8mm_kernel
    grid = (
        (args.m // args.block_m,)
        if args.serial_n
        else (args.m // args.block_m, args.n // args.block_n)
    )

    def run() -> None:
        if args.kai_layout:
            kernel[grid](
                packed_lhs_q,
                packed_lhs_scale,
                packed,
                packed_rhs_scale,
                output,
                M=args.m,
                N=args.n,
                K=args.k,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                PACKED_ABI=args.kai_abi,
            )
        else:
            kernel[grid](
                packed_lhs_q,
                packed_lhs_scale,
                packed,
                packed_rhs_scale,
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
        dot = (
            lhs[:, begin : begin + 32].to(torch.int32)
            @ weight[:, begin : begin + 32].to(torch.int32).T
        )
        expected += (
            dot.float()
            * lhs_scale[:, group].float()[:, None]
            * rhs_scale[group].float()[None, :]
        )
    torch.testing.assert_close(output, expected, rtol=3.0e-5, atol=3.0e-5)

    compiled = last_compiled(kernel)
    smmla, sdot, zip1, stack, asm_lines = audit_codegen(compiled)
    if args.kai_layout and zip1:
        raise RuntimeError(
            f"KAI-panel lowering retained {zip1} nibble ZIP1 shuffles"
        )
    for _ in range(args.warmup):
        run()
    begin = time.perf_counter_ns()
    for _ in range(args.iters):
        run()
    elapsed_us = (time.perf_counter_ns() - begin) / args.iters / 1000.0
    print(
        f"PASS direct-layout Q4-prefill M={args.m} N={args.n} K={args.k} "
        f"BM={args.block_m} BN={args.block_n} serial_n={args.serial_n} "
        f"m4_schedule={args.m4_schedule} kai_layout={args.kai_layout} "
        f"kai_abi={args.kai_abi}\n"
        f"python_launch_us={elapsed_us:.3f}\n"
        f"smmla={smmla}\n"
        f"sdot={sdot}\n"
        f"zip1={zip1}\n"
        f"stack_load_store={stack}\n"
        f"asm_lines={asm_lines}"
    )


if __name__ == "__main__":
    main()
