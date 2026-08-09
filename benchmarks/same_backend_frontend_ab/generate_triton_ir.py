#!/usr/bin/env python3
"""Emit pre-LLVM-O3 IR from shape-specialized ordinary Triton kernels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl

from bench_w8_kleidiai_layout_codegen import (
    _kai_w8_layout_pointer_kernel,
    _kai_w8_layout_outer_pointer_kernel,
)


@triton.jit
def _rms_same_backend_triton(
    input_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STORE_BLOCK_N: tl.constexpr,
):
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for offset in range(0, N, STORE_BLOCK_N):
        cols = offset + tl.arange(0, STORE_BLOCK_N)
        value = tl.load(input_ptr + cols).to(tl.float32)
        sum_sq += tl.sum(value * value, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / N + EPS)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        value = tl.load(input_ptr + cols).to(tl.float32)
        weight = tl.load(weight_ptr + cols)
        normalized = (value * rrms).to(tl.bfloat16)
        tl.store(output_ptr + cols, (normalized * weight).to(tl.bfloat16))


@triton.jit
def _rope_same_backend_triton(
    q_ptr,
    k_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    head = tl.program_id(0)
    row = tl.where(
        head < Q_HEADS,
        q_ptr + head * HEAD_DIM,
        k_ptr + (head - Q_HEADS) * HEAD_DIM,
    )
    half: tl.constexpr = HEAD_DIM // 2
    position = tl.load(positions_ptr).to(tl.int64)
    cache = cos_sin_cache_ptr + position * HEAD_DIM
    for offset in range(0, half, BLOCK_HALF):
        cols = offset + tl.arange(0, BLOCK_HALF)
        first = tl.load(row + cols).to(tl.float32)
        second = tl.load(row + half + cols).to(tl.float32)
        cosine = tl.load(cache + cols).to(tl.float32)
        sine = tl.load(cache + half + cols).to(tl.float32)
        tl.store(
            row + cols,
            (first * cosine - second * sine).to(tl.bfloat16),
        )
        tl.store(
            row + half + cols,
            (first * sine + second * cosine).to(tl.bfloat16),
        )


def dump(compiled, stem: str, output: Path) -> None:
    for stage in ("ttir", "ttcir", "tttcir", "llir"):
        if stage in compiled.asm:
            (output / f"{stem}.{stage}").write_text(
                str(compiled.asm[stage])
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    x = torch.empty(1024, dtype=torch.bfloat16)
    weight = torch.empty_like(x)
    out = torch.empty_like(x)
    rms = _rms_same_backend_triton.warmup(
        x,
        weight,
        out,
        N=1024,
        EPS=1.0e-6,
        BLOCK_N=16,
        STORE_BLOCK_N=16,
        grid=(1,),
    )
    dump(rms, "triton_rms_raw", args.output)

    rms_store8 = _rms_same_backend_triton.warmup(
        x,
        weight,
        out,
        N=1024,
        EPS=1.0e-6,
        BLOCK_N=16,
        STORE_BLOCK_N=8,
        grid=(1,),
    )
    dump(rms_store8, "triton_rms_store8_raw", args.output)

    rms_store32 = _rms_same_backend_triton.warmup(
        x,
        weight,
        out,
        N=1024,
        EPS=1.0e-6,
        BLOCK_N=16,
        STORE_BLOCK_N=32,
        grid=(1,),
    )
    dump(rms_store32, "triton_rms_store32_raw", args.output)

    # Keep the reduction tile at 16 elements but use an 8-element output
    # iteration.  This isolates the 32-byte STP/store stride from reduction
    # code generation in the Cortex-A720 page-offset sensitivity audit.
    rms_output8 = _rms_same_backend_triton.warmup(
        x,
        weight,
        out,
        N=1024,
        EPS=1.0e-6,
        BLOCK_N=8,
        STORE_BLOCK_N=16,
        grid=(1,),
    )
    dump(rms_output8, "triton_rms_output8_raw", args.output)

    q = torch.empty((16, 128), dtype=torch.bfloat16)
    k = torch.empty((8, 128), dtype=torch.bfloat16)
    positions = torch.empty(1, dtype=torch.int64)
    cache = torch.empty((32, 128), dtype=torch.bfloat16)
    rope = _rope_same_backend_triton.warmup(
        q,
        k,
        positions,
        cache,
        Q_HEADS=16,
        HEAD_DIM=128,
        BLOCK_HALF=16,
        grid=(24,),
    )
    dump(rope, "triton_rope_raw", args.output)

    lhs = torch.empty(1024 + 8, dtype=torch.int8)
    rhs = torch.empty((3072 // 4) * (4 * 1024 + 48), dtype=torch.int8)
    clamp = torch.empty(2, dtype=torch.float32)
    w8_out = torch.empty(3072, dtype=torch.float32)
    w8 = _kai_w8_layout_pointer_kernel.warmup(
        lhs,
        rhs,
        clamp,
        w8_out,
        0,
        3072 // 4,
        K=1024,
        N=3072,
        UNROLL=4,
        KEEP_K8_SHAPE=True,
        OUTPUT_BF16=False,
        grid=(1,),
    )
    dump(w8, "triton_w8_raw", args.output)

    w8_outer_pointer = _kai_w8_layout_outer_pointer_kernel.warmup(
        lhs,
        rhs,
        clamp,
        w8_out,
        0,
        3072 // 4,
        K=1024,
        N=3072,
        UNROLL=4,
        KEEP_K8_SHAPE=True,
        OUTPUT_BF16=False,
        grid=(1,),
    )
    dump(w8_outer_pointer, "triton_w8_outer_pointer_raw", args.output)


if __name__ == "__main__":
    main()
