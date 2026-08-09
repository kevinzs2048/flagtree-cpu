#!/usr/bin/env python3
"""Ordinary-Triton BF16 pack for KleidiAI's asymmetric qai8dxp ABI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = Path(
    os.getenv(
        "TRITON_CPU_VENV_SITE",
        "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
    )
)
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON",
        ROOT / "ports/triton-cpu-3.7.2/python",
    )
)
sys.path[:0] = [str(TRITON_PYTHON), str(VENV_SITE)]

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.language.extra import libdevice  # noqa: E402


@triton.jit
def _pack_lhs_qai8dxp_bf16_kernel(
    x_ptr,
    packed_ptr,
    M: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    K: tl.constexpr,
    MR: tl.constexpr,
):
    """Quantize finite BF16 rows into KAI's MR1/MR4 qai8dxp layout."""
    row = tl.program_id(0)
    row_ok = row < M
    source_row = tl.minimum(row, M - 1)
    row_base = x_ptr + source_row * STRIDE_XM

    lanes32 = tl.arange(0, 32)
    row_min = tl.zeros((1,), tl.float32)
    row_max = tl.zeros((1,), tl.float32)
    for off in tl.range(0, K, 32, loop_unroll_factor=1):
        values = tl.load(row_base + off + lanes32).to(tl.float32)
        values = tl.where(row_ok, values, 0.0)
        row_min = tl.minimum(row_min, tl.min(values, axis=0))
        row_max = tl.maximum(row_max, tl.max(values, axis=0))

    quant_scale = tl.where(
        row_min == row_max, 1.0, 255.0 / (row_max - row_min)
    )
    reciprocal_scale = tl.where(
        quant_scale != 0.0, 1.0 / quant_scale, 0.0
    )
    descaled_min = row_min * quant_scale
    descaled_max = row_max * quant_scale
    min_error = -128.0 + descaled_min
    max_error = 127.0 + descaled_max
    zero_point = tl.where(
        min_error + max_error > 0.0,
        -128.0 - descaled_min,
        127.0 - descaled_max,
    )
    zero_point = tl.minimum(tl.maximum(zero_point, -128.0), 127.0)
    zero_point_i32 = libdevice.rint(zero_point).to(tl.int32)

    panel = row // MR
    panel_row = row % MR
    panel_stride: tl.constexpr = MR * (K + 8)
    lanes8 = tl.arange(0, 8)
    for off in tl.range(0, K, 8, loop_unroll_factor=1):
        values = tl.load(row_base + off + lanes8).to(tl.float32)
        values = tl.where(row_ok, values, 0.0)
        quantized = (
            libdevice.rint(values * quant_scale).to(tl.int32)
            + zero_point_i32
        )
        quantized = tl.minimum(tl.maximum(quantized, -128), 127).to(
            tl.int8
        )
        packed_offset = (
            panel * panel_stride
            + (off // 8) * MR * 8
            + panel_row * 8
        )
        tl.store(packed_ptr + packed_offset + lanes8, quantized)

    metadata = packed_ptr + panel * panel_stride + MR * K
    offset_ptr = metadata.to(tl.pointer_type(tl.int32))
    scale_ptr = (metadata + MR * 4).to(tl.pointer_type(tl.float32))
    scalar_lane = tl.arange(0, 1)
    tl.store(offset_ptr + panel_row + scalar_lane, -zero_point_i32)
    tl.store(scale_ptr + panel_row + scalar_lane, reciprocal_scale)


@triton.jit
def _pack_lhs_qai8dxp_bf16_mr4_kernel(
    x_ptr,
    packed_ptr,
    M: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    K: tl.constexpr,
    VECTOR_REDUCTION: tl.constexpr = False,
    NARROW_QUANT: tl.constexpr = False,
):
    """Pack one complete MR4 panel per program, matching KAI's work unit."""
    panel = tl.program_id(0)
    rows = panel * 4 + tl.arange(0, 4)
    if VECTOR_REDUCTION:
        # Match KAI's reduction schedule: keep eight independent extrema per
        # row through the K loop and reduce those lanes only once at the end.
        # The zero initialization folds the required min(value, 0) and
        # max(value, 0) into the same loop-carried vectors.
        reduce_lanes = tl.arange(0, 8)
        row_min_lanes = tl.zeros((4, 8), tl.float32)
        row_max_lanes = tl.zeros((4, 8), tl.float32)
        for off in tl.range(0, K, 8, loop_unroll_factor=1):
            values = tl.load(
                x_ptr
                + rows[:, None] * STRIDE_XM
                + off
                + reduce_lanes[None, :]
            ).to(tl.float32)
            row_min_lanes = tl.minimum(row_min_lanes, values)
            row_max_lanes = tl.maximum(row_max_lanes, values)
        row_min = tl.min(row_min_lanes, axis=1)
        row_max = tl.max(row_max_lanes, axis=1)
    else:
        lanes32 = tl.arange(0, 32)
        row_min = tl.zeros((4,), tl.float32)
        row_max = tl.zeros((4,), tl.float32)
        for off in tl.range(0, K, 32, loop_unroll_factor=1):
            values = tl.load(
                x_ptr
                + rows[:, None] * STRIDE_XM
                + off
                + lanes32[None, :]
            ).to(tl.float32)
            row_min = tl.minimum(row_min, tl.min(values, axis=1))
            row_max = tl.maximum(row_max, tl.max(values, axis=1))

    quant_scale = tl.where(
        row_min == row_max, 1.0, 255.0 / (row_max - row_min)
    )
    reciprocal_scale = tl.where(
        quant_scale != 0.0, 1.0 / quant_scale, 0.0
    )
    descaled_min = row_min * quant_scale
    descaled_max = row_max * quant_scale
    min_error = -128.0 + descaled_min
    max_error = 127.0 + descaled_max
    zero_point = tl.where(
        min_error + max_error > 0.0,
        -128.0 - descaled_min,
        127.0 - descaled_max,
    )
    zero_point = tl.minimum(tl.maximum(zero_point, -128.0), 127.0)
    zero_point_i32 = libdevice.rint(zero_point).to(tl.int32)

    lanes8 = tl.arange(0, 8)
    physical_lanes = tl.arange(0, 32)
    panel_stride: tl.constexpr = 4 * (K + 8)
    for off in tl.range(0, K, 8, loop_unroll_factor=1):
        values = tl.load(
            x_ptr
            + rows[:, None] * STRIDE_XM
            + off
            + lanes8[None, :]
        ).to(tl.float32)
        rounded = libdevice.rint(values * quant_scale[:, None]).to(tl.int32)
        if NARROW_QUANT:
            # The asymmetric scale maps every finite row value into
            # [-255, 255], so int16 is exact here.  Expressing KAI's narrow
            # intermediate lets LLVM use halfword add/clamp/narrow operations
            # instead of keeping all eight vectors in int32.
            quantized = rounded.to(tl.int16) + zero_point_i32.to(tl.int16)[
                :, None
            ]
        else:
            quantized = rounded + zero_point_i32[:, None]
        quantized = tl.minimum(tl.maximum(quantized, -128), 127).to(tl.int8)
        tl.store(
            packed_ptr
            + panel * panel_stride
            + (off // 8) * 32
            + physical_lanes,
            quantized.reshape((32,)),
        )

    metadata = packed_ptr + panel * panel_stride + 4 * K
    tl.store(
        metadata.to(tl.pointer_type(tl.int32)) + tl.arange(0, 4),
        -zero_point_i32,
    )
    tl.store(
        (metadata + 16).to(tl.pointer_type(tl.float32)) + tl.arange(0, 4),
        reciprocal_scale,
    )


@triton.jit
def _pack_lhs_qai8dxp_bf16_mr2_kernel(
    x_ptr,
    packed_ptr,
    M: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    K: tl.constexpr,
):
    """Pack two adjacent rows in an MR4 panel with lower register pressure."""
    pair = tl.program_id(0)
    panel = pair // 2
    panel_row = (pair % 2) * 2
    rows = pair * 2 + tl.arange(0, 2)
    lanes32 = tl.arange(0, 32)
    row_min = tl.zeros((2,), tl.float32)
    row_max = tl.zeros((2,), tl.float32)
    for off in tl.range(0, K, 32, loop_unroll_factor=1):
        values = tl.load(
            x_ptr
            + rows[:, None] * STRIDE_XM
            + off
            + lanes32[None, :]
        ).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(values, axis=1))
        row_max = tl.maximum(row_max, tl.max(values, axis=1))

    quant_scale = tl.where(
        row_min == row_max, 1.0, 255.0 / (row_max - row_min)
    )
    reciprocal_scale = tl.where(
        quant_scale != 0.0, 1.0 / quant_scale, 0.0
    )
    descaled_min = row_min * quant_scale
    descaled_max = row_max * quant_scale
    min_error = -128.0 + descaled_min
    max_error = 127.0 + descaled_max
    zero_point = tl.where(
        min_error + max_error > 0.0,
        -128.0 - descaled_min,
        127.0 - descaled_max,
    )
    zero_point = tl.minimum(tl.maximum(zero_point, -128.0), 127.0)
    zero_point_i32 = libdevice.rint(zero_point).to(tl.int32)

    lanes8 = tl.arange(0, 8)
    physical_lanes = tl.arange(0, 16)
    panel_stride: tl.constexpr = 4 * (K + 8)
    for off in tl.range(0, K, 8, loop_unroll_factor=1):
        values = tl.load(
            x_ptr
            + rows[:, None] * STRIDE_XM
            + off
            + lanes8[None, :]
        ).to(tl.float32)
        quantized = (
            libdevice.rint(values * quant_scale[:, None]).to(tl.int32)
            + zero_point_i32[:, None]
        )
        quantized = tl.minimum(tl.maximum(quantized, -128), 127).to(
            tl.int8
        )
        tl.store(
            packed_ptr
            + panel * panel_stride
            + (off // 8) * 32
            + panel_row * 8
            + physical_lanes,
            quantized.reshape((16,)),
        )

    metadata = packed_ptr + panel * panel_stride + 4 * K
    tl.store(
        metadata.to(tl.pointer_type(tl.int32))
        + panel_row
        + tl.arange(0, 2),
        -zero_point_i32,
    )
    tl.store(
        (metadata + 16).to(tl.pointer_type(tl.float32))
        + panel_row
        + tl.arange(0, 2),
        reciprocal_scale,
    )


def packed_size(m: int, k: int, mr: int) -> int:
    return ((m + mr - 1) // mr) * mr * (k + 8)


def python_reference(x: torch.Tensor, mr: int) -> torch.Tensor:
    m, k = x.shape
    padded_m = ((m + mr - 1) // mr) * mr
    packed = torch.empty(padded_m * (k + 8), dtype=torch.int8)
    for row in range(padded_m):
        values = x[min(row, m - 1)].float() if row < m else torch.zeros(k)
        row_min = torch.minimum(values.min(), torch.tensor(0.0))
        row_max = torch.maximum(values.max(), torch.tensor(0.0))
        quant_scale = torch.where(
            row_min == row_max,
            torch.tensor(1.0),
            torch.tensor(255.0) / (row_max - row_min),
        )
        reciprocal = torch.where(
            quant_scale != 0.0,
            torch.tensor(1.0) / quant_scale,
            torch.tensor(0.0),
        )
        descaled_min = row_min * quant_scale
        descaled_max = row_max * quant_scale
        min_error = -128.0 + descaled_min
        max_error = 127.0 + descaled_max
        zero = torch.where(
            min_error + max_error > 0.0,
            -128.0 - descaled_min,
            127.0 - descaled_max,
        ).clamp(-128.0, 127.0)
        zero_i32 = torch.round(zero).to(torch.int32)
        quantized = (
            torch.round(values * quant_scale).to(torch.int32) + zero_i32
        ).clamp(-128, 127).to(torch.int8)

        panel = row // mr
        panel_row = row % mr
        stride = mr * (k + 8)
        panel_base = panel * stride
        for off in range(0, k, 8):
            begin = panel_base + (off // 8) * mr * 8 + panel_row * 8
            packed[begin : begin + 8].copy_(quantized[off : off + 8])
        meta = panel_base + mr * k
        packed[meta : meta + 4 * mr].view(torch.int32)[panel_row] = -zero_i32
        packed[meta + 4 * mr : meta + 8 * mr].view(torch.float32)[
            panel_row
        ] = reciprocal
    return packed


def audit(compiled) -> dict[str, int]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    return {
        "asm_lines": len(assembly.splitlines()),
        "fcvtns": assembly.count("fcvtns"),
        "stack_load_store": sum(
            "[sp" in line and line.lstrip().startswith(("ld", "st"))
            for line in assembly.splitlines()
        ),
        "external_calls": sum(
            " call " in line and "llvm." not in line
            for line in llir.splitlines()
        ),
        "residual_dot": llir.count("triton_cpu.dot"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--mr", type=int, choices=(1, 4), default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--schedule", choices=("row", "panel2", "panel4"), default="row"
    )
    parser.add_argument("--vector-reduction", action="store_true")
    parser.add_argument("--narrow-quant", action="store_true")
    args = parser.parse_args()
    if args.m <= 0 or args.k <= 0 or args.k % 32:
        raise ValueError("requires M>0 and K%32=0")
    if args.m > 1 and args.m % args.mr:
        raise ValueError("multi-row validation requires M%MR=0")
    if args.schedule != "row" and (args.mr != 4 or args.m % 4):
        raise ValueError("panel schedules require MR=4 and M%4=0")
    if args.vector_reduction and args.schedule != "panel4":
        raise ValueError("vector reduction is currently a panel4 experiment")
    if args.narrow_quant and args.schedule != "panel4":
        raise ValueError("narrow quantization is currently a panel4 experiment")

    torch.manual_seed(8407)
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    packed = torch.empty(packed_size(args.m, args.k, args.mr), dtype=torch.int8)
    if args.schedule != "row":
        if args.schedule == "panel4":
            kernel = _pack_lhs_qai8dxp_bf16_mr4_kernel
            grid = (args.m // 4,)
        else:
            kernel = _pack_lhs_qai8dxp_bf16_mr2_kernel
            grid = (args.m // 2,)
        if args.schedule == "panel4":
            compiled = kernel.warmup(
                x,
                packed,
                M=args.m,
                STRIDE_XM=x.stride(0),
                K=args.k,
                VECTOR_REDUCTION=args.vector_reduction,
                NARROW_QUANT=args.narrow_quant,
                grid=grid,
            )
        else:
            compiled = kernel.warmup(
                x,
                packed,
                M=args.m,
                STRIDE_XM=x.stride(0),
                K=args.k,
                grid=grid,
            )
    else:
        kernel = _pack_lhs_qai8dxp_bf16_kernel
        grid = (((args.m + args.mr - 1) // args.mr) * args.mr,)
        compiled = kernel.warmup(
            x,
            packed,
            M=args.m,
            STRIDE_XM=x.stride(0),
            K=args.k,
            MR=args.mr,
            grid=grid,
        )
    codegen = audit(compiled)
    if codegen["external_calls"] or codegen["residual_dot"]:
        raise RuntimeError(f"bad generated pack: {codegen}")
    if args.compile_only:
        print(json.dumps({"status": "COMPILED", **codegen}, indent=2))
        return

    if args.schedule != "row":
        if args.schedule == "panel4":
            kernel[grid](
                x,
                packed,
                M=args.m,
                STRIDE_XM=x.stride(0),
                K=args.k,
                VECTOR_REDUCTION=args.vector_reduction,
                NARROW_QUANT=args.narrow_quant,
            )
        else:
            kernel[grid](
                x,
                packed,
                M=args.m,
                STRIDE_XM=x.stride(0),
                K=args.k,
            )
    else:
        kernel[grid](
            x,
            packed,
            M=args.m,
            STRIDE_XM=x.stride(0),
            K=args.k,
            MR=args.mr,
        )
    expected = python_reference(x, args.mr)
    if not torch.equal(packed, expected):
        mismatch = torch.nonzero(packed != expected).flatten()
        raise AssertionError(
            f"packed blob mismatch count={mismatch.numel()} "
            f"first={mismatch[:16].tolist()}"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "m": args.m,
                "k": args.k,
                "mr": args.mr,
                "schedule": args.schedule,
                "vector_reduction": args.vector_reduction,
                "narrow_quant": args.narrow_quant,
                "packed_bytes": packed.numel(),
                "python_reference_bit_exact": True,
                **codegen,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
