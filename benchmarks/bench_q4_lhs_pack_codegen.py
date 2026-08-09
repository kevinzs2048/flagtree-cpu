#!/usr/bin/env python3
"""A/B the production M4 scale-vectorized Q4 pack and retired baselines."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
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
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from flag_gems.runtime.backend._arm.q4.kernels import (  # noqa: E402
    _pack_lhs_qsi8d32p_panel4_scalar_kernel,
    _pack_lhs_qsi8d32p_row_kernel,
    _q4_lhs_full_row_absmax,
    _q4_lhs_quantize_full_row_pair,
    _round_to_nearest_even,
)


@triton.jit
def _wide_pack_reference(
    x_ptr,
    lhs_scale_ptr,
    lhs_data_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    row_ok = row < M
    panel = row // 4
    panel_row = row % 4
    lanes = tl.arange(0, 32)
    groups: tl.constexpr = K // 32
    data_offsets = (lanes // 8) * 32 + panel_row * 8 + (lanes % 8)
    for group in tl.range(0, groups, loop_unroll_factor=1):
        values = tl.load(
            x_ptr + row * stride_xm + group * 32 + lanes,
            mask=row_ok,
            other=0.0,
        ).to(tl.float32)
        absmax = tl.max(tl.abs(values), axis=0)
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        scaled = tl.minimum(tl.maximum(values * inv_scale, -127.0), 127.0)
        quantized = _round_to_nearest_even(scaled).to(tl.int8)
        blob_base = (panel * groups + group) * 136
        tl.store(
            lhs_scale_ptr + blob_base // 2 + panel_row,
            scale.to(tl.float16),
        )
        tl.store(
            lhs_data_ptr + blob_base + 8 + data_offsets,
            quantized,
        )


@triton.jit
def _panel4_interleaved_reduction(
    x_ptr,
    lhs_scale_ptr,
    lhs_data_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Test an M4-parallel K8 reduction without changing the production path."""
    panel = tl.program_id(0)
    groups: tl.constexpr = K // 32
    row0 = panel * 4
    rows = row0 + tl.arange(0, 4)
    lanes = tl.arange(0, 8)
    for group in tl.range(0, groups, loop_unroll_factor=1):
        offsets = (
            rows[:, None] * stride_xm
            + group * 32
            + lanes[None, :]
        )
        values0 = tl.load(x_ptr + offsets).to(tl.float32)
        values1 = tl.load(x_ptr + offsets + 8).to(tl.float32)
        values2 = tl.load(x_ptr + offsets + 16).to(tl.float32)
        values3 = tl.load(x_ptr + offsets + 24).to(tl.float32)
        max0 = tl.max(tl.abs(values0), axis=1)
        max1 = tl.max(tl.abs(values1), axis=1)
        max2 = tl.max(tl.abs(values2), axis=1)
        max3 = tl.max(tl.abs(values3), axis=1)
        absmax = tl.maximum(
            tl.maximum(max0, max1), tl.maximum(max2, max3)
        )
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        blob_base = (panel * groups + group) * 136
        tl.store(
            lhs_scale_ptr + blob_base // 2 + tl.arange(0, 4),
            scale.to(tl.float16),
        )
        inv_even, inv_odd = tl.split(inv_scale.reshape((2, 2)))
        inv0, inv2 = tl.split(inv_even)
        inv1, inv3 = tl.split(inv_odd)
        x0 = x_ptr + row0 * stride_xm + group * 32
        x1 = x0 + stride_xm
        x2 = x1 + stride_xm
        x3 = x2 + stride_xm
        data_base = lhs_data_ptr + blob_base + 8
        _q4_lhs_quantize_full_row_pair(x0, x1, data_base, inv0, inv1)
        _q4_lhs_quantize_full_row_pair(
            x2, x3, data_base + 16, inv2, inv3
        )


@triton.jit
def _quantize_symmetric_i8_unclamped(values, inv_scale):
    """RNE conversion without redundant saturation for absmax scaling."""
    return _round_to_nearest_even(values * inv_scale).to(tl.int8)


@triton.jit
def _q4_lhs_quantize_full_row_pair_unclamped(
    x0_base,
    x1_base,
    data_base,
    inv0,
    inv1,
):
    lanes = tl.arange(0, 8)
    store_lanes = tl.arange(0, 16)
    for offset in tl.static_range(0, 32, 8):
        values0 = tl.load(x0_base + offset + lanes).to(tl.float32)
        values1 = tl.load(x1_base + offset + lanes).to(tl.float32)
        quant0 = _quantize_symmetric_i8_unclamped(values0, inv0)
        quant1 = _quantize_symmetric_i8_unclamped(values1, inv1)
        quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
        tl.store(data_base + offset * 4 + store_lanes, quant01)


@triton.jit
def _q4_lhs_quantize_full_panel4_unclamped(
    x0_base,
    x1_base,
    x2_base,
    x3_base,
    data_base,
    inv0,
    inv1,
    inv2,
    inv3,
):
    lanes = tl.arange(0, 8)
    store_lanes = tl.arange(0, 32)
    for offset in tl.static_range(0, 32, 8):
        values0 = tl.load(x0_base + offset + lanes).to(tl.float32)
        values1 = tl.load(x1_base + offset + lanes).to(tl.float32)
        values2 = tl.load(x2_base + offset + lanes).to(tl.float32)
        values3 = tl.load(x3_base + offset + lanes).to(tl.float32)
        quant0 = _quantize_symmetric_i8_unclamped(values0, inv0)
        quant1 = _quantize_symmetric_i8_unclamped(values1, inv1)
        quant2 = _quantize_symmetric_i8_unclamped(values2, inv2)
        quant3 = _quantize_symmetric_i8_unclamped(values3, inv3)
        quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
        quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
        quant0123 = (
            tl.join(quant01, quant23).permute(1, 0).reshape((32,))
        )
        tl.store(data_base + offset * 4 + store_lanes, quant0123)


@triton.jit
def _q4_lhs_full_row_absmax_bits(x_base):
    """Find a finite BF16 absmax in its monotonic unsigned encoding."""
    lanes = tl.arange(0, 8)
    values0 = tl.load(x_base + lanes)
    values1 = tl.load(x_base + 8 + lanes)
    values2 = tl.load(x_base + 16 + lanes)
    values3 = tl.load(x_base + 24 + lanes)
    bits0 = (values0.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits1 = (values1.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits2 = (values2.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits3 = (values3.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    max01 = tl.where(bits0 > bits1, bits0, bits1)
    max23 = tl.where(bits2 > bits3, bits2, bits3)
    lane_max = tl.where(max01 > max23, max01, max23)
    absmax_bits = tl.max(lane_max, axis=0)
    return (
        absmax_bits.to(tl.uint16)
        .to(tl.bfloat16, bitcast=True)
        .to(tl.float32)
    )


@triton.jit
def _panel4_scalar_unclamped(
    x_ptr,
    lhs_scale_ptr,
    lhs_data_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Production layout with saturation removed under the absmax invariant."""
    panel = tl.program_id(0)
    row0 = panel * 4
    groups: tl.constexpr = K // 32
    for group in tl.range(0, groups, loop_unroll_factor=1):
        x0 = x_ptr + row0 * stride_xm + group * 32
        x1 = x0 + stride_xm
        x2 = x1 + stride_xm
        x3 = x2 + stride_xm
        max0 = _q4_lhs_full_row_absmax(x0)
        max1 = _q4_lhs_full_row_absmax(x1)
        max2 = _q4_lhs_full_row_absmax(x2)
        max3 = _q4_lhs_full_row_absmax(x3)
        absmax = tl.join(
            tl.join(max0, max2), tl.join(max1, max3)
        ).reshape((4,))
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        blob_base = (panel * groups + group) * 136
        tl.store(
            lhs_scale_ptr + blob_base // 2 + tl.arange(0, 4),
            scale.to(tl.float16),
        )
        inv_even, inv_odd = tl.split(inv_scale.reshape((2, 2)))
        inv0, inv2 = tl.split(inv_even)
        inv1, inv3 = tl.split(inv_odd)
        data_base = lhs_data_ptr + blob_base + 8
        _q4_lhs_quantize_full_panel4_unclamped(
            x0,
            x1,
            x2,
            x3,
            data_base,
            inv0,
            inv1,
            inv2,
            inv3,
        )


@triton.jit
def _panel4_scalar_bitmax_unclamped(
    x_ptr,
    lhs_scale_ptr,
    lhs_data_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Use finite BF16 bit ordering for the reduction, then direct RNE."""
    panel = tl.program_id(0)
    row0 = panel * 4
    groups: tl.constexpr = K // 32
    for group in tl.range(0, groups, loop_unroll_factor=1):
        x0 = x_ptr + row0 * stride_xm + group * 32
        x1 = x0 + stride_xm
        x2 = x1 + stride_xm
        x3 = x2 + stride_xm
        max0 = _q4_lhs_full_row_absmax_bits(x0)
        max1 = _q4_lhs_full_row_absmax_bits(x1)
        max2 = _q4_lhs_full_row_absmax_bits(x2)
        max3 = _q4_lhs_full_row_absmax_bits(x3)
        absmax = tl.join(
            tl.join(max0, max2), tl.join(max1, max3)
        ).reshape((4,))
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        blob_base = (panel * groups + group) * 136
        tl.store(
            lhs_scale_ptr + blob_base // 2 + tl.arange(0, 4),
            scale.to(tl.float16),
        )
        inv_even, inv_odd = tl.split(inv_scale.reshape((2, 2)))
        inv0, inv2 = tl.split(inv_even)
        inv1, inv3 = tl.split(inv_odd)
        data_base = lhs_data_ptr + blob_base + 8
        _q4_lhs_quantize_full_panel4_unclamped(
            x0,
            x1,
            x2,
            x3,
            data_base,
            inv0,
            inv1,
            inv2,
            inv3,
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


def audit(compiled) -> dict[str, object]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    return {
        "asm_lines": len(assembly.splitlines()),
        "stack_frame": [
            line.strip()
            for line in assembly.splitlines()
            if "sub\tsp" in line or "sub sp" in line
        ][:1],
        "folded_spills": assembly.count("folded spill"),
        "folded_reloads": assembly.count("folded reload"),
        "external_calls": sum(
            " call " in line and "llvm." not in line and " asm " not in line
            for line in llir.splitlines()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=9)
    args = parser.parse_args()
    if args.m <= 0 or args.k % 32:
        raise ValueError("requires M>0 and K%32=0")

    torch.set_num_threads(1)
    torch.manual_seed(4207)
    padded_m = 4 * ((args.m + 3) // 4)
    groups = args.k // 32
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    wide = torch.empty((padded_m // 4) * groups * 136, dtype=torch.uint8)
    bounded = torch.empty_like(wide)
    panel4_scalar = torch.empty_like(wide)
    interleaved = torch.empty_like(wide)
    unclamped = torch.empty_like(wide)
    bitmax = torch.empty_like(wide)
    grid = (padded_m,)

    def run_wide():
        _wide_pack_reference[grid](
            x,
            wide.view(torch.float16),
            wide.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
        )

    def run_bounded():
        _pack_lhs_qsi8d32p_row_kernel[grid](
            x,
            bounded.view(torch.float16),
            bounded.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
        )

    def run_panel4_scalar():
        _pack_lhs_qsi8d32p_panel4_scalar_kernel[(padded_m // 4,)](
            x,
            panel4_scalar.view(torch.float16),
            panel4_scalar.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            FULL_PANEL=args.m % 4 == 0,
            num_warps=1,
            num_stages=1,
        )

    def run_interleaved():
        _panel4_interleaved_reduction[(padded_m // 4,)](
            x,
            interleaved.view(torch.float16),
            interleaved.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
        )

    def run_unclamped():
        _panel4_scalar_unclamped[(padded_m // 4,)](
            x,
            unclamped.view(torch.float16),
            unclamped.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
        )

    def run_bitmax():
        _panel4_scalar_bitmax_unclamped[(padded_m // 4,)](
            x,
            bitmax.view(torch.float16),
            bitmax.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
        )

    run_wide()
    run_bounded()
    run_panel4_scalar()
    if args.m % 4 == 0:
        run_interleaved()
        run_unclamped()
        run_bitmax()
    assert torch.equal(wide, bounded)
    assert torch.equal(wide, panel4_scalar)
    if args.m % 4 == 0:
        assert torch.equal(wide, interleaved)
        assert torch.equal(wide, unclamped)
        assert torch.equal(wide, bitmax)
    wide_compiled = _wide_pack_reference.warmup(
        x,
        wide.view(torch.float16),
        wide.view(torch.int8),
        args.m,
        x.stride(0),
        K=args.k,
        num_warps=1,
        num_stages=1,
        grid=grid,
    )
    bounded_compiled = _pack_lhs_qsi8d32p_row_kernel.warmup(
        x,
        bounded.view(torch.float16),
        bounded.view(torch.int8),
        args.m,
        x.stride(0),
        K=args.k,
        num_warps=1,
        num_stages=1,
        grid=grid,
    )
    panel4_scalar_compiled = _pack_lhs_qsi8d32p_panel4_scalar_kernel.warmup(
        x,
        panel4_scalar.view(torch.float16),
        panel4_scalar.view(torch.int8),
        args.m,
        x.stride(0),
        K=args.k,
        FULL_PANEL=args.m % 4 == 0,
        num_warps=1,
        num_stages=1,
        grid=(padded_m // 4,),
    )
    interleaved_compiled = None
    if args.m % 4 == 0:
        interleaved_compiled = _panel4_interleaved_reduction.warmup(
            x,
            interleaved.view(torch.float16),
            interleaved.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
            grid=(padded_m // 4,),
        )
    unclamped_compiled = None
    if args.m % 4 == 0:
        unclamped_compiled = _panel4_scalar_unclamped.warmup(
            x,
            unclamped.view(torch.float16),
            unclamped.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
            grid=(padded_m // 4,),
        )
    bitmax_compiled = None
    if args.m % 4 == 0:
        bitmax_compiled = _panel4_scalar_bitmax_unclamped.warmup(
            x,
            bitmax.view(torch.float16),
            bitmax.view(torch.int8),
            args.m,
            x.stride(0),
            K=args.k,
            num_warps=1,
            num_stages=1,
            grid=(padded_m // 4,),
        )
    wide_us = median_us(run_wide, args.warmup, args.iters, args.batches)
    bounded_us = median_us(
        run_bounded, args.warmup, args.iters, args.batches
    )
    panel4_scalar_us = median_us(
        run_panel4_scalar, args.warmup, args.iters, args.batches
    )
    interleaved_us = (
        median_us(run_interleaved, args.warmup, args.iters, args.batches)
        if interleaved_compiled is not None
        else None
    )
    unclamped_us = (
        median_us(run_unclamped, args.warmup, args.iters, args.batches)
        if unclamped_compiled is not None
        else None
    )
    bitmax_us = (
        median_us(run_bitmax, args.warmup, args.iters, args.batches)
        if bitmax_compiled is not None
        else None
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "triton_module": triton.__file__,
                "m": args.m,
                "k": args.k,
                "bit_exact_blob": True,
                "wide_us": wide_us,
                "bounded_us": bounded_us,
                "panel4_scalar_us": panel4_scalar_us,
                "interleaved_us": interleaved_us,
                "unclamped_us": unclamped_us,
                "bitmax_us": bitmax_us,
                "bounded_over_wide": bounded_us / wide_us,
                "panel4_scalar_over_bounded": panel4_scalar_us / bounded_us,
                "wide_codegen": audit(wide_compiled),
                "bounded_codegen": audit(bounded_compiled),
                "panel4_scalar_codegen": audit(panel4_scalar_compiled),
                "interleaved_codegen": (
                    audit(interleaved_compiled)
                    if interleaved_compiled is not None
                    else None
                ),
                "unclamped_codegen": (
                    audit(unclamped_compiled)
                    if unclamped_compiled is not None
                    else None
                ),
                "bitmax_codegen": (
                    audit(bitmax_compiled)
                    if bitmax_compiled is not None
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
