#!/usr/bin/env python3
"""Prototype a true packed-Q4 x grouped-Q8 GEMV through ordinary Triton.

The packed weight layout is compiler-oriented but lossless relative to Q4_0:

    [N/4, K/32, 4, 4, 4] uint8

The final two dimensions are [output lane, K lane], so every contiguous
16-byte load is already in SDOT order.  For every byte, the low nibble holds
weight k and the high nibble holds weight k+16.  Scales are
[N/4, K/32, 4].  The Triton kernel keeps weights packed in memory, unpacks
nibbles in registers and expresses the matrix product as two ordinary
``tl.dot`` operations per 32-element group.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a8_grouped_gemv_kernel(
    x_q_ptr,
    x_scale_ptr,
    w_q4_ptr,
    w_scale_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    WHOLE_PROJECTION: tl.constexpr,
):
    group_count: tl.constexpr = K // 32
    cols = tl.arange(0, 4)
    n2d = cols[None, :]
    if WHOLE_PROJECTION:
        tile_begin = range_begin
        tile_end = range_end
    else:
        tile_begin = tl.program_id(0)
        tile_end = tile_begin + 1

    for tile in range(tile_begin, tile_end):
        result = tl.zeros((1, 4), dtype=tl.float32)
        for group in range(0, group_count):
            x_base = group * 32
            dot = tl.zeros((1, 4), dtype=tl.int32)
            # Stream four packed K rows at a time.  This preserves ordinary
            # tl.load/bitwise/tl.dot semantics while exposing the lifetime a
            # fused nibble-unpack + SDOT compiler lowering should produce.
            for kb in range(0, 16, 4):
                k4 = kb + tl.arange(0, 4)
                packed_flat = tl.load(
                    w_q4_ptr
                    + ((tile * group_count + group) * 4 + kb // 4) * 16
                    + tl.arange(0, 16)
                )
                # Memory is [N lane, K lane]; tl.dot consumes [K, N].
                packed = tl.trans(packed_flat.reshape((4, 4)))
                weight_lo = (
                    (packed & 0x0F).to(tl.int8) - 8
                ).to(tl.int8)
                weight_hi = (
                    (packed >> 4).to(tl.int8) - 8
                ).to(tl.int8)

                x_lo = tl.load(
                    x_q_ptr + x_base + k4
                ).reshape((1, 4))
                x_hi = tl.load(
                    x_q_ptr + x_base + 16 + k4
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
            ).to(tl.float32)
            x_scale = tl.load(x_scale_ptr + group)
            result += (
                dot.to(tl.float32)
                * weight_scale[None, :]
                * x_scale
            )

        output_offset = tile * 4 + cols
        tl.store(
            out_ptr + output_offset, result.reshape((4,))
        )


@triton.jit
def _w4a8_grouped_gemv_static_kernel(
    x_q_ptr,
    x_scale_ptr,
    w_q4_ptr,
    w_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # A separate whole-projection entry point lets LLVM constant-fold both
    # loop bounds.  The range entry remains available for threadpool work
    # stealing without imposing its dynamic-loop overhead on one worker.
    _w4a8_grouped_gemv_kernel(
        x_q_ptr,
        x_scale_ptr,
        w_q4_ptr,
        w_scale_ptr,
        out_ptr,
        0,
        N // 4,
        K=K,
        N=N,
        TILE_N=TILE_N,
        WHOLE_PROJECTION=True,
    )


def pack_q4_microtiles(
    weight_q4: torch.Tensor, weight_scale: torch.Tensor, tile_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack signed [-8,7] [K,N] weights without expanding their 4-bit data."""
    k, n = weight_q4.shape
    if tile_n != 4 or k % 32 or n % 4:
        raise ValueError("dot-ready Q4 pack requires TILE_N=4, K%32=0, N%4=0")
    groups = k // 32
    grouped = weight_q4.reshape(groups, 32, n).to(torch.int16)
    low = (grouped[:, :16, :] + 8).to(torch.uint8)
    high = (grouped[:, 16:, :] + 8).to(torch.uint8)
    packed = low | (high << 4)
    packed = (
        packed.reshape(groups, 4, 4, n // 4, 4)
        .permute(3, 0, 1, 4, 2)
        .contiguous()
    )
    scales = (
        weight_scale.reshape(groups, n // 4, 4)
        .permute(1, 0, 2)
        .contiguous()
    )
    return packed, scales


def reference_grouped_w4a8(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight_q4: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    k, n = weight_q4.shape
    result = torch.zeros(n, dtype=torch.float32)
    for group in range(k // 32):
        begin = group * 32
        dot = torch._int_mm(
            x_q[begin : begin + 32].reshape(1, 32),
            weight_q4[begin : begin + 32],
        ).reshape(-1)
        result += (
            dot.float()
            * x_scale[group]
            * weight_scale[group]
        )
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--tile-n", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--grid",
        choices=("whole", "tiles"),
        default="whole",
    )
    parser.add_argument(
        "--static-whole",
        action="store_true",
        help="compile a constant-bound whole-projection AOT entry point",
    )
    parser.add_argument(
        "--c-lib",
        type=str,
        default="",
        help="optional .so containing the matching w4a8_microtile_c ceiling",
    )
    args = parser.parse_args()

    torch.manual_seed(401)
    x_q = torch.randint(-127, 128, (args.k,), dtype=torch.int8)
    x_scale = torch.rand(args.k // 32, dtype=torch.float32) / 127.0
    weight_q4 = torch.randint(
        -8, 8, (args.k, args.n), dtype=torch.int8
    )
    weight_scale = (
        torch.rand(args.k // 32, args.n, dtype=torch.float32) / 7.0
    )
    packed, packed_scale = pack_q4_microtiles(
        weight_q4, weight_scale, args.tile_n
    )
    output = torch.empty(args.n, dtype=torch.float32)

    def run() -> None:
        if args.static_whole:
            _w4a8_grouped_gemv_static_kernel[(1,)](
                x_q,
                x_scale,
                packed,
                packed_scale,
                output,
                K=args.k,
                N=args.n,
                TILE_N=args.tile_n,
            )
        else:
            grid = (1,) if args.grid == "whole" else (args.n // 4,)
            _w4a8_grouped_gemv_kernel[grid](
                x_q,
                x_scale,
                packed,
                packed_scale,
                output,
                0,
                args.n // 4,
                K=args.k,
                N=args.n,
                TILE_N=args.tile_n,
                WHOLE_PROJECTION=args.grid == "whole",
            )

    run()
    reference = reference_grouped_w4a8(
        x_q, x_scale, weight_q4, weight_scale
    )
    delta = (output - reference).abs()
    if not torch.allclose(output, reference, rtol=1.0e-6, atol=1.2e-5):
        raise AssertionError(
            "W4A8 mismatch: "
            f"count={torch.count_nonzero(delta).item()} "
            f"max={delta.max().item()}"
        )

    latency = median_us(run, args.warmup, args.iters, args.batches)
    compiled_kernel = (
        _w4a8_grouped_gemv_static_kernel
        if args.static_whole
        else _w4a8_grouped_gemv_kernel
    )
    cache = next(iter(compiled_kernel.device_caches.values()))[0]
    compiled = list(cache.values())[-1]
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"]
    c_report = ""
    if args.c_lib:
        library = ctypes.CDLL(args.c_lib)
        c_kernel = library.w4a8_microtile_c
        c_kernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        c_output = torch.empty_like(output)

        def run_c() -> None:
            c_kernel(
                x_q.data_ptr(),
                x_scale.data_ptr(),
                packed.data_ptr(),
                packed_scale.data_ptr(),
                c_output.data_ptr(),
                args.k,
                args.n,
            )

        run_c()
        c_delta = (c_output - reference).abs()
        if not torch.allclose(c_output, reference, rtol=1.0e-6, atol=1.2e-5):
            raise AssertionError(
                "C W4A8 mismatch: "
                f"count={torch.count_nonzero(c_delta).item()} "
                f"max={c_delta.max().item()}"
            )
        c_latency = median_us(run_c, args.warmup, args.iters, args.batches)
        c_report = (
            f"\nc_microtile_us={c_latency:.3f}"
            f"\ntriton_over_c={latency / c_latency:.3f}x"
            f"\nc_fp32_max_abs={c_delta.max().item():.8f}"
        )

    external_calls = [
        line
        for line in llir.splitlines()
        if " call " in line
        and "llvm." not in line
        and "_w4a8_grouped_gemv_kernel" not in line
    ]
    print(
        f"PASS K={args.k} N={args.n} TILE_N={args.tile_n} "
        f"GRID={args.grid}\n"
        f"triton_kernel_us={latency:.3f}\n"
        f"fp32_exact={torch.equal(output, reference)}\n"
        f"fp32_max_abs={delta.max().item():.8f}\n"
        f"packed_weight_bytes={packed.numel()}\n"
        f"int8_weight_bytes={weight_q4.numel()}\n"
        f"compression={weight_q4.numel() / packed.numel():.1f}x\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"asm_smmla={assembly.count('smmla')}\n"
        f"llir_sdot={llir.count('llvm.aarch64.neon.sdot')}\n"
        f"external_runtime_calls={len(external_calls)}"
        f"{c_report}"
    )


if __name__ == "__main__":
    main()
