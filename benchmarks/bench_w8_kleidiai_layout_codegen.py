#!/usr/bin/env python3
"""Ordinary-Triton GEMV over KleidiAI's qai8dxp/qsi8cxp blobs.

This is intentionally an ABI/layout experiment rather than a replacement
format.  The kernel consumes the exact byte organization expected by
KleidiAI's 1x4 SDOT W8 microkernel, including the LHS zero-point/scale and
the RHS sums/scales/bias epilogue data.
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _kai_w8_layout_gemv_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    # qai8dxp: K signed bytes, then int32(-zero_point), then float scale.
    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    rhs_stride: tl.constexpr = 4 * K + 48
    rhs_values_bytes: tl.constexpr = 4 * K
    lanes = tl.arange(0, BLOCK_N * 4)
    local_col = lanes // 4
    local_k = lanes % 4
    out_lanes = tl.arange(0, BLOCK_N)

    for block in range(range_begin, range_end):
        dot = tl.zeros((1, BLOCK_N), dtype=tl.int32)
        global_col = block * BLOCK_N + local_col
        rhs_block = global_col // 4
        rhs_lane = global_col % 4

        for group in tl.range(0, K // 4, loop_unroll_factor=UNROLL):
            logical_k = group * 4 + local_k
            # qsi8cxp4x8 stores [K/8, 4 outputs, 8 K bytes].
            rhs_byte = (
                rhs_block * rhs_stride
                + (logical_k // 8) * 32
                + rhs_lane * 8
                + logical_k % 8
            )
            packed = tl.load(rhs_packed_ptr + rhs_byte)
            weight = tl.trans(packed.reshape((BLOCK_N, 4)))
            x = tl.load(
                lhs_packed_ptr + group * 4 + tl.arange(0, 4)
            ).reshape((1, 4))
            dot += tl.dot(x, weight, out_dtype=tl.int32)

        epilogue_col = block * BLOCK_N + out_lanes
        epilogue_block = epilogue_col // 4
        epilogue_lane = epilogue_col % 4
        epilogue_base = epilogue_block * rhs_stride + rhs_values_bytes
        rhs_sum = tl.load(
            (rhs_packed_ptr + epilogue_base + epilogue_lane * 4).to(
                tl.pointer_type(tl.int32)
            )
        )
        rhs_scale = tl.load(
            (rhs_packed_ptr + epilogue_base + 16 + epilogue_lane * 4).to(
                tl.pointer_type(tl.float32)
            )
        )
        bias = tl.load(
            (rhs_packed_ptr + epilogue_base + 32 + epilogue_lane * 4).to(
                tl.pointer_type(tl.float32)
            )
        )
        corrected = dot.reshape((BLOCK_N,)) + rhs_sum * lhs_offset
        result = corrected.to(tl.float32) * lhs_scale * rhs_scale + bias
        result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
        tl.store(out_ptr + epilogue_col, result)


@triton.jit
def _kai_w8_layout_reduce_kernel(
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
    """Keep KAI's contiguous 4-output x 8-K microtile intact.

    This is ordinary Triton integer multiply/reduce code.  It deliberately
    avoids reconstructing the logical KxN matrix, giving LLVM a chance to
    recognize the four adjacent 8-byte dot products.
    """
    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    rhs_stride: tl.constexpr = 4 * K + 48
    q_offsets = tl.arange(0, 32)
    output_lanes = tl.arange(0, 4)
    k_lanes = tl.arange(0, 8)

    for block in range(range_begin, range_end):
        dot = tl.zeros((4,), dtype=tl.int32)
        rhs_base = block * rhs_stride
        for group in tl.range(0, K // 8, loop_unroll_factor=UNROLL):
            weight = tl.load(
                rhs_packed_ptr + rhs_base + group * 32 + q_offsets
            ).reshape((4, 8))
            x = tl.load(lhs_packed_ptr + group * 8 + k_lanes).reshape(
                (1, 8)
            )
            dot += tl.sum(weight.to(tl.int32) * x.to(tl.int32), axis=1)

        epilogue_base = rhs_base + 4 * K
        rhs_sum = tl.load(
            (rhs_packed_ptr + epilogue_base + output_lanes * 4).to(
                tl.pointer_type(tl.int32)
            )
        )
        rhs_scale = tl.load(
            (rhs_packed_ptr + epilogue_base + 16 + output_lanes * 4).to(
                tl.pointer_type(tl.float32)
            )
        )
        bias = tl.load(
            (rhs_packed_ptr + epilogue_base + 32 + output_lanes * 4).to(
                tl.pointer_type(tl.float32)
            )
        )
        corrected = dot + rhs_sum * lhs_offset
        result = corrected.to(tl.float32) * lhs_scale * rhs_scale + bias
        result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
        tl.store(out_ptr + block * 4 + output_lanes, result)


@triton.jit
def _kai_w8_layout_partial_kernel(
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
    """Mirror KAI's two partial accumulators per output in ordinary Triton."""
    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    rhs_stride: tl.constexpr = 4 * K + 48
    q_offsets = tl.arange(0, 32)
    output_lanes = tl.arange(0, 4)
    k_lanes = tl.arange(0, 8)

    for block in range(range_begin, range_end):
        # KAI keeps the low/high four-element dots separate over the full K
        # loop, then horizontally reduces once in the epilogue.
        partial = tl.zeros((4, 2), dtype=tl.int32)
        rhs_base = block * rhs_stride
        for group in tl.range(0, K // 8, loop_unroll_factor=UNROLL):
            weight = tl.load(
                rhs_packed_ptr + rhs_base + group * 32 + q_offsets
            ).reshape((4, 2, 4))
            x = tl.load(lhs_packed_ptr + group * 8 + k_lanes).reshape(
                (1, 2, 4)
            )
            partial += tl.sum(
                weight.to(tl.int32) * x.to(tl.int32), axis=2
            )

        dot = tl.sum(partial, axis=1)
        epilogue_base = rhs_base + 4 * K
        rhs_sum = tl.load(
            (rhs_packed_ptr + epilogue_base + output_lanes * 4).to(
                tl.pointer_type(tl.int32)
            )
        )
        rhs_scale = tl.load(
            (rhs_packed_ptr + epilogue_base + 16 + output_lanes * 4).to(
                tl.pointer_type(tl.float32)
            )
        )
        bias = tl.load(
            (rhs_packed_ptr + epilogue_base + 32 + output_lanes * 4).to(
                tl.pointer_type(tl.float32)
            )
        )
        corrected = dot + rhs_sum * lhs_offset
        result = corrected.to(tl.float32) * lhs_scale * rhs_scale + bias
        result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
        tl.store(out_ptr + block * 4 + output_lanes, result)


@triton.jit
def _kai_w8_layout_split_kernel(
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
    """KAI physical layout with two native one-dimensional accumulators."""
    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    rhs_stride: tl.constexpr = 4 * K + 48
    q_offsets = tl.arange(0, 16)
    output_lanes = tl.arange(0, 4)
    x_offsets = tl.arange(0, 8)

    for block in range(range_begin, range_end):
        partial01 = tl.zeros((4,), dtype=tl.int32)
        partial23 = tl.zeros((4,), dtype=tl.int32)
        rhs_base = block * rhs_stride
        # KAI schedules four K8 microsteps as one K32 loop body.  Express
        # that structure directly instead of relying on the generic loop
        # unroller to rediscover pointer post-increments.
        for chunk in tl.range(0, K // 32, loop_unroll_factor=UNROLL):
            for sub in tl.static_range(0, 4):
                group = chunk * 4 + sub
                group_base = rhs_base + group * 32
                weight01 = tl.load(
                    rhs_packed_ptr + group_base + q_offsets
                ).reshape((4, 4))
                weight23 = tl.load(
                    rhs_packed_ptr + group_base + 16 + q_offsets
                ).reshape((4, 4))
                x = tl.load(
                    lhs_packed_ptr + group * 8 + x_offsets
                ).reshape((2, 4))
                # [x0..3, x4..7, x0..3, x4..7], matching the two
                # four-byte partial lanes for each adjacent output pair.
                # Preserve the 2x4 structure through join.  This is
                # semantically the same repeated 8-byte vector as flattening
                # first, but gives Arm LLVM a 64-bit broadcast and selects one
                # LD1R rather than LDP(D) plus two lane MOVs.
                x_repeated = tl.join(x, x).permute(2, 0, 1).reshape((4, 4))
                partial01 += tl.sum(
                    weight01.to(tl.int32) * x_repeated.to(tl.int32),
                    axis=1,
                )
                partial23 += tl.sum(
                    weight23.to(tl.int32) * x_repeated.to(tl.int32),
                    axis=1,
                )

        # A single 4x2 pair reduction maps to ADDP(partial01, partial23).
        partial = tl.join(partial01, partial23).permute(
            1, 0
        ).reshape((4, 2))
        dot = tl.sum(partial, axis=1)
        epilogue_base = rhs_base + 4 * K
        rhs_sum_base = (rhs_packed_ptr + epilogue_base).to(
            tl.pointer_type(tl.int32)
        )
        rhs_sum = tl.load(
            rhs_sum_base + output_lanes
        )
        rhs_scale_base = (rhs_packed_ptr + epilogue_base + 16).to(
            tl.pointer_type(tl.float32)
        )
        rhs_scale = tl.load(
            rhs_scale_base + output_lanes
        )
        bias_base = (rhs_packed_ptr + epilogue_base + 32).to(
            tl.pointer_type(tl.float32)
        )
        bias = tl.load(
            bias_base + output_lanes
        )
        corrected = dot + rhs_sum * lhs_offset
        result = corrected.to(tl.float32) * lhs_scale * rhs_scale + bias
        result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
        tl.store(out_ptr + block * 4 + output_lanes, result)


@triton.jit
def _kai_w8_layout_pointer_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    KEEP_K8_SHAPE: tl.constexpr = True,
    OUTPUT_BF16: tl.constexpr = False,
):
    """Split accumulators with explicit K-loop pointer induction."""
    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    rhs_stride: tl.constexpr = 4 * K + 48
    q_offsets = tl.arange(0, 16)
    output_lanes = tl.arange(0, 4)
    x_offsets = tl.arange(0, 8)

    for block in range(range_begin, range_end):
        partial01 = tl.zeros((4,), dtype=tl.int32)
        partial23 = tl.zeros((4,), dtype=tl.int32)
        rhs_tile_ptr = rhs_packed_ptr + block * rhs_stride
        lhs_chunk_ptr = lhs_packed_ptr
        rhs_chunk_ptr = rhs_tile_ptr
        for _ in tl.range(0, K // 32, loop_unroll_factor=UNROLL):
            for sub in tl.static_range(0, 4):
                group_ptr = rhs_chunk_ptr + sub * 32
                weight01 = tl.load(
                    group_ptr + q_offsets
                ).reshape((4, 4))
                weight23 = tl.load(
                    group_ptr + 16 + q_offsets
                ).reshape((4, 4))
                x = tl.load(
                    lhs_chunk_ptr + sub * 8 + x_offsets
                ).reshape((2, 4))
                if KEEP_K8_SHAPE:
                    # Keep the physical K8 value intact so the Arm backend
                    # can broadcast it with LD1R. Flatten-before-join
                    # obscures that operation and expands to LDP(D) plus
                    # lane insertion MOVs.
                    x_repeated = tl.join(x, x).permute(2, 0, 1).reshape(
                        (4, 4)
                    )
                else:
                    # Reproducible retired A/B graph. This is never selected
                    # by auto; it exists to make the instruction regression
                    # independently measurable from a fresh cache.
                    x_flat = x.reshape((8,))
                    x_repeated = tl.join(x_flat, x_flat).permute(
                        1, 0
                    ).reshape((4, 4))
                partial01 += tl.sum(
                    weight01.to(tl.int32) * x_repeated.to(tl.int32),
                    axis=1,
                )
                partial23 += tl.sum(
                    weight23.to(tl.int32) * x_repeated.to(tl.int32),
                    axis=1,
                )
            lhs_chunk_ptr += 32
            rhs_chunk_ptr += 128

        partial = tl.join(partial01, partial23).permute(
            1, 0
        ).reshape((4, 2))
        dot = tl.sum(partial, axis=1)
        epilogue_base = (rhs_tile_ptr + 4 * K).to(
            tl.pointer_type(tl.int32)
        )
        rhs_sum = tl.load(epilogue_base + output_lanes)
        rhs_scale_base = (rhs_tile_ptr + 4 * K + 16).to(
            tl.pointer_type(tl.float32)
        )
        rhs_scale = tl.load(rhs_scale_base + output_lanes)
        bias_base = (rhs_tile_ptr + 4 * K + 32).to(
            tl.pointer_type(tl.float32)
        )
        bias = tl.load(bias_base + output_lanes)
        corrected = dot + rhs_sum * lhs_offset
        result = corrected.to(tl.float32) * lhs_scale * rhs_scale + bias
        result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
        if OUTPUT_BF16:
            result = result.to(tl.bfloat16)
        tl.store(out_ptr + block * 4 + output_lanes, result)


@triton.jit
def _kai_w8_layout_outer_pointer_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    KEEP_K8_SHAPE: tl.constexpr = True,
    OUTPUT_BF16: tl.constexpr = False,
):
    """KAI N4 kernel carrying panel/output pointers through the outer loop."""
    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    rhs_stride: tl.constexpr = 4 * K + 48
    q_offsets = tl.arange(0, 16)
    output_lanes = tl.arange(0, 4)
    x_offsets = tl.arange(0, 8)
    rhs_tile_ptr = rhs_packed_ptr + range_begin * rhs_stride
    output_tile_ptr = out_ptr + range_begin * 4

    for _block in range(range_begin, range_end):
        partial01 = tl.zeros((4,), dtype=tl.int32)
        partial23 = tl.zeros((4,), dtype=tl.int32)
        lhs_chunk_ptr = lhs_packed_ptr
        rhs_chunk_ptr = rhs_tile_ptr
        for _ in tl.range(0, K // 32, loop_unroll_factor=UNROLL):
            for sub in tl.static_range(0, 4):
                group_ptr = rhs_chunk_ptr + sub * 32
                weight01 = tl.load(
                    group_ptr + q_offsets
                ).reshape((4, 4))
                weight23 = tl.load(
                    group_ptr + 16 + q_offsets
                ).reshape((4, 4))
                x = tl.load(
                    lhs_chunk_ptr + sub * 8 + x_offsets
                ).reshape((2, 4))
                if KEEP_K8_SHAPE:
                    x_repeated = tl.join(x, x).permute(
                        2, 0, 1
                    ).reshape((4, 4))
                else:
                    x_flat = x.reshape((8,))
                    x_repeated = tl.join(x_flat, x_flat).permute(
                        1, 0
                    ).reshape((4, 4))
                partial01 += tl.sum(
                    weight01.to(tl.int32) * x_repeated.to(tl.int32),
                    axis=1,
                )
                partial23 += tl.sum(
                    weight23.to(tl.int32) * x_repeated.to(tl.int32),
                    axis=1,
                )
            lhs_chunk_ptr += 32
            rhs_chunk_ptr += 128

        partial = tl.join(partial01, partial23).permute(
            1, 0
        ).reshape((4, 2))
        dot = tl.sum(partial, axis=1)
        epilogue_base = (rhs_tile_ptr + 4 * K).to(
            tl.pointer_type(tl.int32)
        )
        rhs_sum = tl.load(epilogue_base + output_lanes)
        rhs_scale_base = (rhs_tile_ptr + 4 * K + 16).to(
            tl.pointer_type(tl.float32)
        )
        rhs_scale = tl.load(rhs_scale_base + output_lanes)
        bias_base = (rhs_tile_ptr + 4 * K + 32).to(
            tl.pointer_type(tl.float32)
        )
        bias = tl.load(bias_base + output_lanes)
        corrected = dot + rhs_sum * lhs_offset
        result = corrected.to(tl.float32) * lhs_scale * rhs_scale + bias
        result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
        if OUTPUT_BF16:
            result = result.to(tl.bfloat16)
        tl.store(output_tile_ptr + output_lanes, result)
        rhs_tile_ptr += rhs_stride
        output_tile_ptr += 4


@triton.jit
def _kai_w8_layout_pointer_bn8_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    KEEP_K8_SHAPE: tl.constexpr = True,
    OUTPUT_BF16: tl.constexpr = False,
):
    """Two adjacent KAI N4 panels sharing each activation K8 load."""
    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    rhs_stride: tl.constexpr = 4 * K + 48
    q_offsets = tl.arange(0, 16)
    output_lanes = tl.arange(0, 4)
    x_offsets = tl.arange(0, 8)

    for block in range(range_begin, range_end):
        partial0_01 = tl.zeros((4,), dtype=tl.int32)
        partial0_23 = tl.zeros((4,), dtype=tl.int32)
        partial1_01 = tl.zeros((4,), dtype=tl.int32)
        partial1_23 = tl.zeros((4,), dtype=tl.int32)
        rhs_tile0_ptr = rhs_packed_ptr + block * 2 * rhs_stride
        rhs_tile1_ptr = rhs_tile0_ptr + rhs_stride
        lhs_chunk_ptr = lhs_packed_ptr
        rhs_chunk0_ptr = rhs_tile0_ptr
        rhs_chunk1_ptr = rhs_tile1_ptr
        for _ in tl.range(0, K // 32, loop_unroll_factor=UNROLL):
            for sub in tl.static_range(0, 4):
                group0_ptr = rhs_chunk0_ptr + sub * 32
                group1_ptr = rhs_chunk1_ptr + sub * 32
                weight0_01 = tl.load(
                    group0_ptr + q_offsets
                ).reshape((4, 4))
                weight0_23 = tl.load(
                    group0_ptr + 16 + q_offsets
                ).reshape((4, 4))
                weight1_01 = tl.load(
                    group1_ptr + q_offsets
                ).reshape((4, 4))
                weight1_23 = tl.load(
                    group1_ptr + 16 + q_offsets
                ).reshape((4, 4))
                x = tl.load(
                    lhs_chunk_ptr + sub * 8 + x_offsets
                ).reshape((2, 4))
                if KEEP_K8_SHAPE:
                    x_repeated = tl.join(x, x).permute(2, 0, 1).reshape(
                        (4, 4)
                    )
                else:
                    x_flat = x.reshape((8,))
                    x_repeated = tl.join(x_flat, x_flat).permute(
                        1, 0
                    ).reshape((4, 4))
                x_i32 = x_repeated.to(tl.int32)
                partial0_01 += tl.sum(
                    weight0_01.to(tl.int32) * x_i32, axis=1
                )
                partial0_23 += tl.sum(
                    weight0_23.to(tl.int32) * x_i32, axis=1
                )
                partial1_01 += tl.sum(
                    weight1_01.to(tl.int32) * x_i32, axis=1
                )
                partial1_23 += tl.sum(
                    weight1_23.to(tl.int32) * x_i32, axis=1
                )
            lhs_chunk_ptr += 32
            rhs_chunk0_ptr += 128
            rhs_chunk1_ptr += 128

        partial0 = tl.join(partial0_01, partial0_23).permute(
            1, 0
        ).reshape((4, 2))
        partial1 = tl.join(partial1_01, partial1_23).permute(
            1, 0
        ).reshape((4, 2))
        dot0 = tl.sum(partial0, axis=1)
        dot1 = tl.sum(partial1, axis=1)

        epilogue0 = (rhs_tile0_ptr + 4 * K).to(
            tl.pointer_type(tl.int32)
        )
        epilogue1 = (rhs_tile1_ptr + 4 * K).to(
            tl.pointer_type(tl.int32)
        )
        rhs_sum0 = tl.load(epilogue0 + output_lanes)
        rhs_sum1 = tl.load(epilogue1 + output_lanes)
        rhs_scale0 = tl.load(
            (rhs_tile0_ptr + 4 * K + 16).to(tl.pointer_type(tl.float32))
            + output_lanes
        )
        rhs_scale1 = tl.load(
            (rhs_tile1_ptr + 4 * K + 16).to(tl.pointer_type(tl.float32))
            + output_lanes
        )
        bias0 = tl.load(
            (rhs_tile0_ptr + 4 * K + 32).to(tl.pointer_type(tl.float32))
            + output_lanes
        )
        bias1 = tl.load(
            (rhs_tile1_ptr + 4 * K + 32).to(tl.pointer_type(tl.float32))
            + output_lanes
        )
        corrected0 = dot0 + rhs_sum0 * lhs_offset
        corrected1 = dot1 + rhs_sum1 * lhs_offset
        result0 = corrected0.to(tl.float32) * lhs_scale * rhs_scale0 + bias0
        result1 = corrected1.to(tl.float32) * lhs_scale * rhs_scale1 + bias1
        result0 = tl.minimum(tl.maximum(result0, clamp_min), clamp_max)
        result1 = tl.minimum(tl.maximum(result1, clamp_min), clamp_max)
        if OUTPUT_BF16:
            result0 = result0.to(tl.bfloat16)
            result1 = result1.to(tl.bfloat16)
        output_base = block * 8
        tl.store(out_ptr + output_base + output_lanes, result0)
        tl.store(out_ptr + output_base + 4 + output_lanes, result1)


def pack_lhs(xq: torch.Tensor, offset: int, scale: float) -> torch.Tensor:
    k = xq.numel()
    packed = torch.empty(k + 8, dtype=torch.int8)
    packed[:k].copy_(xq)
    packed[k : k + 4].view(torch.int32)[0] = offset
    packed[k + 4 : k + 8].view(torch.float32)[0] = scale
    return packed


def pack_rhs(
    weight_nk: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    n, k = weight_nk.shape
    if n % 4 or k % 32:
        raise ValueError("KAI W8 layout requires N%4=0 and K%32=0")
    stride = 4 * k + 48
    packed = torch.zeros((n // 4, stride), dtype=torch.int8)
    values = (
        weight_nk.reshape(n // 4, 4, k // 8, 8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(n // 4, 4 * k)
    )
    packed[:, : 4 * k].copy_(values)
    packed[:, 4 * k : 4 * k + 16].view(torch.int32).copy_(
        weight_nk.to(torch.int32).sum(dim=1).reshape(n // 4, 4)
    )
    packed[:, 4 * k + 16 : 4 * k + 32].view(torch.float32).copy_(
        scale.reshape(n // 4, 4)
    )
    packed[:, 4 * k + 32 : 4 * k + 48].view(torch.float32).copy_(
        bias.reshape(n // 4, 4)
    )
    return packed.reshape(-1)


def last_compiled(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    return list(cache.values())[-1]


def audit_split_pointer_codegen(
    assembly: str, llir: str, mode: str, unroll: int
) -> dict[str, int]:
    """Gate the instruction shape that makes the exact KAI ABI competitive."""
    lines = assembly.splitlines()
    stats = {
        "sdot": assembly.count("sdot"),
        "addp": assembly.count("addp"),
        "ld1r": assembly.count("ld1r"),
        "stack_load_store": sum(
            "[sp" in line
            and line.lstrip().startswith(("ld", "st"))
            for line in lines
        ),
        "lane_insert_moves": sum(
            "mov" in line and ".d[" in line for line in lines
        ),
        "external_calls": sum(
            " call " in line and "llvm." not in line
            for line in llir.splitlines()
        ),
        "residual_dot": llir.count("triton_cpu.dot"),
    }
    if mode not in (
        "split", "pointer", "outer-pointer", "pointer8"
    ):
        return stats
    panels = 2 if mode == "pointer8" else 1
    expected_sdot = panels * 8 * unroll
    expected_ld1r = 2 + 4 * unroll
    if stats["sdot"] != expected_sdot:
        raise RuntimeError(f"{mode} W8 codegen lost expected SDOTs")
    if stats["addp"] != panels:
        raise RuntimeError(f"{mode} W8 codegen lost expected ADDP")
    if stats["ld1r"] != expected_ld1r:
        raise RuntimeError(
            f"{mode} W8 codegen lost activation broadcasts: "
            f"{stats['ld1r']} != {expected_ld1r}"
        )
    if stats["stack_load_store"]:
        raise RuntimeError(f"{mode} W8 codegen introduced stack traffic")
    if stats["lane_insert_moves"]:
        raise RuntimeError(f"{mode} W8 codegen reintroduced lane inserts")
    if stats["external_calls"] or stats["residual_dot"]:
        raise RuntimeError(f"{mode} W8 codegen retained an external/dot call")
    return stats


def select_kai_w8_codegen_kernel(k: int, mode: str):
    """Select the spill-free KAI-layout schedule from shape alone.

    Explicit pointer induction is the stable schedule across the tested
    K=256..9728 range.  It keeps the K8 activation broadcast and RHS stream
    as loop-carried pointers, while preserving the exact packed ABI and
    arithmetic contract of the split-index form.
    """
    if mode == "auto":
        mode = "pointer"
    return mode, {
        "reduce": _kai_w8_layout_reduce_kernel,
        "partial": _kai_w8_layout_partial_kernel,
        "split": _kai_w8_layout_split_kernel,
        "pointer": _kai_w8_layout_pointer_kernel,
        "outer-pointer": _kai_w8_layout_outer_pointer_kernel,
        "pointer8": _kai_w8_layout_pointer_bn8_kernel,
    }.get(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--block-n", type=int, default=4)
    parser.add_argument("--unroll", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=(
            "auto", "dot", "reduce", "partial", "split", "pointer",
            "outer-pointer", "pointer8",
        ),
        default="auto",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--flatten-k8",
        action="store_true",
        help="compile the retired pointer graph for a strict source A/B",
    )
    parser.add_argument(
        "--output-bf16",
        action="store_true",
        help="store the production vLLM BF16 result",
    )
    args = parser.parse_args()
    if args.k % 32 or args.n % args.block_n or args.block_n % 4:
        raise ValueError("requires K%32=0 and N%BLOCK_N=0 and BLOCK_N%4=0")
    if args.flatten_k8 and args.mode not in (
        "auto", "pointer", "outer-pointer", "pointer8"
    ):
        raise ValueError("--flatten-k8 is only valid for pointer modes")
    if args.output_bf16 and args.mode not in (
        "auto", "pointer", "outer-pointer", "pointer8"
    ):
        raise ValueError("--output-bf16 is only valid for pointer modes")
    if args.mode == "pointer8" and args.block_n != 8:
        raise ValueError("pointer8 requires --block-n 8")

    clamp = torch.tensor([-math.inf, math.inf], dtype=torch.float32)
    if args.compile_only:
        # The KAI ABI stores signed activation and weight bytes.  Pointer
        # signedness controls whether lowering emits extsi or extui, so the
        # compile-only placeholder must preserve the real int8 contract.
        byte_buffer = torch.empty(1, dtype=torch.int8)
        output = torch.empty(
            1,
            dtype=torch.bfloat16 if args.output_bf16 else torch.float32,
        )
        common_args = (
            byte_buffer,
            byte_buffer,
            clamp,
            output,
            0,
            args.n // args.block_n,
        )
        selected_mode, selected_kernel = select_kai_w8_codegen_kernel(
            args.k, args.mode
        )
        if selected_mode == "dot":
            kernel = _kai_w8_layout_gemv_kernel
            compiled = kernel.warmup(
                *common_args,
                K=args.k,
                N=args.n,
                BLOCK_N=args.block_n,
                UNROLL=args.unroll,
                grid=(1,),
            )
        else:
            expected_block_n = 8 if selected_mode == "pointer8" else 4
            if args.block_n != expected_block_n:
                raise ValueError(
                    f"{args.mode} mode requires BLOCK_N={expected_block_n}"
                )
            kernel = selected_kernel
            if selected_mode in (
                "pointer", "outer-pointer", "pointer8"
            ):
                compiled = kernel.warmup(
                    *common_args,
                    K=args.k,
                    N=args.n,
                    UNROLL=args.unroll,
                    KEEP_K8_SHAPE=not args.flatten_k8,
                    OUTPUT_BF16=args.output_bf16,
                    grid=(1,),
                )
            else:
                compiled = kernel.warmup(
                    *common_args,
                    K=args.k,
                    N=args.n,
                    UNROLL=args.unroll,
                    grid=(1,),
                )
        assembly = compiled.asm["asm"].lower()
        llir = compiled.asm["llir"].lower()
        audit_mode = (
            "pointer-flat"
            if selected_mode == "pointer" and args.flatten_k8
            else selected_mode
        )
        stats = audit_split_pointer_codegen(
            assembly, llir, audit_mode, args.unroll
        )
        print(
            f"COMPILED KAI-layout W8 requested={args.mode} "
            f"mode={audit_mode} K={args.k} "
            f"N={args.n} BN={args.block_n} UNROLL={args.unroll}\n"
            f"asm_lines={len(assembly.splitlines())}\n"
            f"asm_sdot={stats['sdot']}\n"
            f"asm_addp={stats['addp']}\n"
            f"asm_ld1r={stats['ld1r']}\n"
            f"stack_load_store={stats['stack_load_store']}\n"
            f"lane_insert_moves={stats['lane_insert_moves']}\n"
            f"external_calls={stats['external_calls']}\n"
            f"residual_dot={stats['residual_dot']}"
        )
        return

    torch.manual_seed(1248)
    xq = torch.randint(-128, 128, (args.k,), dtype=torch.int8)
    weight = torch.randint(-127, 128, (args.n, args.k), dtype=torch.int8)
    rhs_scale = torch.rand(args.n, dtype=torch.float32) * 0.01 + 0.001
    bias = torch.rand(args.n, dtype=torch.float32) - 0.5
    lhs_offset = -7
    lhs_scale = 0.013
    lhs_packed = pack_lhs(xq, lhs_offset, lhs_scale)
    rhs_packed = pack_rhs(weight, rhs_scale, bias)
    out = torch.empty(
        args.n,
        dtype=torch.bfloat16 if args.output_bf16 else torch.float32,
    )

    selected_mode, selected_kernel = select_kai_w8_codegen_kernel(
        args.k, args.mode
    )
    if selected_mode == "dot":
        kernel = _kai_w8_layout_gemv_kernel
        kernel[(1,)](
            lhs_packed,
            rhs_packed,
            clamp,
            out,
            0,
            args.n // args.block_n,
            K=args.k,
            N=args.n,
            BLOCK_N=args.block_n,
            UNROLL=args.unroll,
        )
    elif selected_mode == "reduce":
        if args.block_n != 4:
            raise ValueError("reduce mode directly implements KAI's BLOCK_N=4")
        kernel = _kai_w8_layout_reduce_kernel
        kernel[(1,)](
            lhs_packed,
            rhs_packed,
            clamp,
            out,
            0,
            args.n // 4,
            K=args.k,
            N=args.n,
            UNROLL=args.unroll,
        )
    elif selected_mode == "partial":
        if args.block_n != 4:
            raise ValueError("partial mode directly implements KAI's BLOCK_N=4")
        kernel = _kai_w8_layout_partial_kernel
        kernel[(1,)](
            lhs_packed,
            rhs_packed,
            clamp,
            out,
            0,
            args.n // 4,
            K=args.k,
            N=args.n,
            UNROLL=args.unroll,
        )
    else:
        expected_block_n = 8 if selected_mode == "pointer8" else 4
        if args.block_n != expected_block_n:
            raise ValueError(
                f"{args.mode} mode requires BLOCK_N={expected_block_n}"
            )
        kernel = selected_kernel
        if selected_mode in (
            "pointer", "outer-pointer", "pointer8"
        ):
            kernel[(1,)](
                lhs_packed,
                rhs_packed,
                clamp,
                out,
                0,
                args.n // args.block_n,
                K=args.k,
                N=args.n,
                UNROLL=args.unroll,
                KEEP_K8_SHAPE=not args.flatten_k8,
                OUTPUT_BF16=args.output_bf16,
            )
        else:
            kernel[(1,)](
                lhs_packed,
                rhs_packed,
                clamp,
                out,
                0,
                args.n // 4,
                K=args.k,
                N=args.n,
                UNROLL=args.unroll,
            )
    expected_i32 = (
        xq.to(torch.int32) @ weight.to(torch.int32).T
        + weight.to(torch.int32).sum(dim=1) * lhs_offset
    )
    expected = expected_i32.float() * lhs_scale * rhs_scale + bias
    if args.output_bf16:
        if not torch.equal(out, expected.to(torch.bfloat16)):
            raise AssertionError("BF16 output is not bit-exact")
    else:
        torch.testing.assert_close(out, expected, rtol=1.0e-6, atol=1.0e-6)

    compiled = last_compiled(kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    stats = audit_split_pointer_codegen(
        assembly,
        llir,
        "pointer-flat"
        if selected_mode == "pointer" and args.flatten_k8
        else selected_mode,
        args.unroll,
    )
    print(
        f"PASS KAI-layout W8 requested={args.mode} mode={selected_mode} "
        f"K={args.k} N={args.n} "
        f"BN={args.block_n} UNROLL={args.unroll}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={stats['sdot']}\n"
        f"asm_addp={stats['addp']}\n"
        f"asm_ld1r={stats['ld1r']}\n"
        f"stack_load_store={stats['stack_load_store']}\n"
        f"lane_insert_moves={stats['lane_insert_moves']}\n"
        f"external_calls={stats['external_calls']}\n"
        f"residual_dot={stats['residual_dot']}"
    )


if __name__ == "__main__":
    main()
