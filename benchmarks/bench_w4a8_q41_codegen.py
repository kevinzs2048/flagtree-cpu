#!/usr/bin/env python3
"""Ordinary-Triton dot-ready Q4_1 x Q8_1 decode GEMV.

Q4_1 dequantizes a block as ``d * q + m``.  The SDOT path uses signed
``q - 8`` nibbles and applies the algebraically equivalent correction
``(m + 8*d) * (d_x * sum(q_x))`` once per 32-element group.
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from bench_w4a8_codegen import median_us


@triton.jit
def _w4a8_q4_1_gemv_kernel(
    x_q_ptr,
    x_scale_ptr,
    x_sum_ptr,
    w_q4_ptr,
    w_scale_ptr,
    w_offset_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    WHOLE_PROJECTION: tl.constexpr,
):
    group_count: tl.constexpr = K // 32
    cols = tl.arange(0, 4)
    if WHOLE_PROJECTION:
        tile_begin = range_begin
        tile_end = range_end
    else:
        tile_begin = tl.program_id(0)
        tile_end = tile_begin + 1

    for tile in range(tile_begin, tile_end):
        result = tl.zeros((1, 4), dtype=tl.float32)
        for group in range(0, group_count):
            dot = tl.zeros((1, 4), dtype=tl.int32)
            for kb in range(0, 16, 4):
                packed_flat = tl.load(
                    w_q4_ptr
                    + ((tile * group_count + group) * 4 + kb // 4) * 16
                    + tl.arange(0, 16)
                )
                packed = tl.trans(packed_flat.reshape((4, 4)))
                weight_lo = (
                    (packed & 0x0F).to(tl.int8) - 8
                ).to(tl.int8)
                weight_hi = (
                    (packed >> 4).to(tl.int8) - 8
                ).to(tl.int8)
                k4 = kb + tl.arange(0, 4)
                x_lo = tl.load(
                    x_q_ptr + group * 32 + k4
                ).reshape((1, 4))
                x_hi = tl.load(
                    x_q_ptr + group * 32 + 16 + k4
                ).reshape((1, 4))
                dot += tl.dot(
                    x_lo, weight_lo, out_dtype=tl.int32
                )
                dot += tl.dot(
                    x_hi, weight_hi, out_dtype=tl.int32
                )

            scale_offset = (
                (tile * group_count + group) * 4 + cols
            )
            weight_scale = tl.load(
                w_scale_ptr + scale_offset
            )
            weight_offset = tl.load(w_offset_ptr + scale_offset)
            x_scale = tl.load(x_scale_ptr + group)
            x_sum = tl.load(x_sum_ptr + group)
            result += (
                dot.to(tl.float32)
                * weight_scale[None, :]
                * x_scale
                + weight_offset[None, :] * x_sum
            )

        tl.store(
            out_ptr + tile * 4 + cols,
            result.reshape((4,)),
        )


@triton.jit
def _w4a8_q4_1_gemv_static_kernel(
    x_q_ptr,
    x_scale_ptr,
    x_sum_ptr,
    w_q4_ptr,
    w_scale_ptr,
    w_offset_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    _w4a8_q4_1_gemv_kernel(
        x_q_ptr,
        x_scale_ptr,
        x_sum_ptr,
        w_q4_ptr,
        w_scale_ptr,
        w_offset_ptr,
        out_ptr,
        0,
        N // 4,
        K=K,
        N=N,
        WHOLE_PROJECTION=True,
    )


def pack_q4_1(
    q: torch.Tensor, scale: torch.Tensor, minimum: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k, n = q.shape
    groups = k // 32
    grouped = q.reshape(groups, 32, n).to(torch.uint8)
    packed = grouped[:, :16, :] | (grouped[:, 16:, :] << 4)
    packed = (
        packed.reshape(groups, 4, 4, n // 4, 4)
        .permute(3, 0, 1, 4, 2)
        .contiguous()
    )
    scales = (
        scale.reshape(groups, n // 4, 4)
        .permute(1, 0, 2)
        .contiguous()
    )
    offsets = (
        (minimum + 8.0 * scale)
        .reshape(groups, n // 4, 4)
        .permute(1, 0, 2)
        .contiguous()
    )
    return packed, scales, offsets


def reference(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    minimum: torch.Tensor,
) -> torch.Tensor:
    k, n = q.shape
    out = torch.zeros(n, dtype=torch.float32)
    for group in range(k // 32):
        begin = group * 32
        dot = torch._int_mm(
            x_q[begin : begin + 32].reshape(1, 32),
            q[begin : begin + 32].to(torch.int8),
        ).reshape(-1)
        out += (
            dot.float() * scale[group] * x_scale[group]
            + minimum[group]
            * x_scale[group]
            * x_q[begin : begin + 32].float().sum()
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=9728)
    parser.add_argument("--n", type=int, default=2560)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--static-whole",
        action="store_true",
        help="compile a constant-bound whole-projection AOT entry point",
    )
    args = parser.parse_args()
    if args.k % 32 or args.n % 4:
        raise ValueError("Q4_1 kernel requires K%32=0 and N%4=0")

    torch.manual_seed(411)
    x_q = torch.randint(-127, 128, (args.k,), dtype=torch.int8)
    x_scale = torch.rand(args.k // 32, dtype=torch.float32) / 127.0
    x_sum = (
        x_q.reshape(-1, 32).float().sum(1) * x_scale
    ).contiguous()
    q = torch.randint(0, 16, (args.k, args.n), dtype=torch.uint8)
    weight_scale = (
        torch.rand(args.k // 32, args.n, dtype=torch.float32) / 15.0
    )
    weight_min = (
        torch.rand(args.k // 32, args.n, dtype=torch.float32) - 0.5
    )
    packed, packed_scale, packed_offset = pack_q4_1(
        q, weight_scale, weight_min
    )
    output = torch.empty(args.n, dtype=torch.float32)

    def run() -> None:
        if args.static_whole:
            _w4a8_q4_1_gemv_static_kernel[(1,)](
                x_q,
                x_scale,
                x_sum,
                packed,
                packed_scale,
                packed_offset,
                output,
                K=args.k,
                N=args.n,
            )
        else:
            _w4a8_q4_1_gemv_kernel[(1,)](
                x_q,
                x_scale,
                x_sum,
                packed,
                packed_scale,
                packed_offset,
                output,
                0,
                args.n // 4,
                K=args.k,
                N=args.n,
                WHOLE_PROJECTION=True,
            )

    run()
    expected = reference(
        x_q, x_scale, q, weight_scale, weight_min
    )
    delta = (output - expected).abs()
    if not torch.allclose(output, expected, rtol=2.0e-6, atol=2.0e-5):
        raise AssertionError(
            f"Q4_1 mismatch max_abs={delta.max().item():.8f}"
        )

    latency = median_us(
        run, args.warmup, args.iters, args.batches
    )
    compiled_kernel = (
        _w4a8_q4_1_gemv_static_kernel
        if args.static_whole
        else _w4a8_q4_1_gemv_kernel
    )
    cache = next(iter(compiled_kernel.device_caches.values()))[0]
    compiled = list(cache.values())[-1]
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"]
    external_calls = [
        line
        for line in llir.splitlines()
        if " call " in line
        and "llvm." not in line
        and "_w4a8_q4_1_gemv_kernel" not in line
    ]
    print(
        f"PASS Q4_1 K={args.k} N={args.n}\n"
        f"triton_kernel_us={latency:.3f}\n"
        f"fp32_max_abs={delta.max().item():.8f}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"llir_sdot={llir.count('llvm.aarch64.neon.sdot')}\n"
        f"external_runtime_calls={len(external_calls)}"
    )


if __name__ == "__main__":
    main()
