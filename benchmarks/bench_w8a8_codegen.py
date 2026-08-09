#!/usr/bin/env python3
"""Packed W8A8 decode GEMV expressed only with ordinary Triton operations.

Weights use the SDOT-ready layout [N/4, K/4, 4, 4].  Each 16-byte tile is
[output lane, K lane], so the kernel only needs a reshape/transposition before
feeding the logical [4,K] values to ``tl.dot``.  No TLE op or runtime compute
function is used.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a8_grouped_gemv_kernel(
    x_q_ptr,
    x_scale_ptr,
    packed_ptr,
    weight_scale_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    OUT_BF16: tl.constexpr,
    WHOLE_PROJECTION: tl.constexpr,
):
    k_groups: tl.constexpr = K // 4
    cols = tl.arange(0, 4)
    if WHOLE_PROJECTION:
        tile_begin = range_begin
        tile_end = range_end
    else:
        tile_begin = tl.program_id(0)
        tile_end = tile_begin + 1

    for tile in range(tile_begin, tile_end):
        dot = tl.zeros((1, 4), dtype=tl.int32)
        for group in tl.range(
            0, k_groups, loop_unroll_factor=UNROLL
        ):
            packed_flat = tl.load(
                packed_ptr + (tile * k_groups + group) * 16
                + tl.arange(0, 16)
            )
            # Packed memory is [N lane, K lane]; tl.dot consumes [K, N].
            weight = tl.trans(packed_flat.reshape((4, 4)))
            x = tl.load(
                x_q_ptr + group * 4 + tl.arange(0, 4)
            ).reshape((1, 4))
            dot += tl.dot(x, weight, out_dtype=tl.int32)

        scale = tl.load(weight_scale_ptr + tile * 4 + cols)
        x_scale = tl.load(x_scale_ptr)
        if OUT_BF16:
            result = dot.to(tl.float32) * x_scale * scale[None, :]
            tl.store(
                out_ptr + tile * 4 + cols,
                result.reshape((4,)).to(tl.bfloat16),
            )
        else:
            result = dot.to(tl.float32) * scale[None, :] * x_scale
            tl.store(out_ptr + tile * 4 + cols, result.reshape((4,)))


def pack_w8_microtiles(weight_kn: torch.Tensor) -> torch.Tensor:
    k, n = weight_kn.shape
    if k % 4 or n % 4:
        raise ValueError("packed W8 layout requires K%4=0 and N%4=0")
    return (
        weight_kn.reshape(k // 4, 4, n // 4, 4)
        .permute(2, 0, 3, 1)
        .contiguous()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--grid", choices=("whole", "tiles"), default="whole")
    parser.add_argument("--unroll", type=int, default=2)
    args = parser.parse_args()
    if args.k % 4 or args.n % 4:
        raise ValueError("W8A8 kernel requires K%4=0 and N%4=0")

    torch.manual_seed(801)
    x_q = torch.randint(-127, 128, (args.k,), dtype=torch.int8)
    x_scale = torch.rand(1, dtype=torch.float32) / 127.0
    weight = torch.randint(-127, 128, (args.k, args.n), dtype=torch.int8)
    weight_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    packed = pack_w8_microtiles(weight)
    output = torch.empty(args.n, dtype=torch.float32)

    def run() -> None:
        grid = (1,) if args.grid == "whole" else (args.n // 4,)
        _w8a8_grouped_gemv_kernel[grid](
            x_q,
            x_scale,
            packed,
            weight_scale,
            output,
            0,
            args.n // 4,
            K=args.k,
            N=args.n,
            UNROLL=args.unroll,
            OUT_BF16=False,
            WHOLE_PROJECTION=args.grid == "whole",
        )

    run()
    expected = (
        (x_q.to(torch.int32) @ weight.to(torch.int32)).float()
        * weight_scale
        * x_scale
    )
    delta = (output - expected).abs()
    if not torch.equal(output, expected):
        raise AssertionError(
            f"W8A8 mismatch count={torch.count_nonzero(delta).item()} "
            f"max_abs={delta.max().item():.8f}"
        )

    latency = median_us(run, args.warmup, args.iters, args.batches)
    cache = next(iter(_w8a8_grouped_gemv_kernel.device_caches.values()))[0]
    compiled = list(cache.values())[-1]
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    external_calls = [
        line
        for line in llir.splitlines()
        if " call " in line and "llvm." not in line and " asm " not in line
    ]
    print(
        f"PASS W8A8 K={args.k} N={args.n} GRID={args.grid}\n"
        f"triton_kernel_us={latency:.3f}\n"
        f"bit_exact={torch.equal(output, expected)}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"llir_sdot={llir.count('llvm.aarch64.neon.sdot')}\n"
        f"external_runtime_calls={len(external_calls)}"
    )


if __name__ == "__main__":
    main()
