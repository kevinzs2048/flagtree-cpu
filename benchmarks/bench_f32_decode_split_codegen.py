#!/usr/bin/env python3
"""Two-stage ggml FP32 W8 decode with shared compiler-generated quantization."""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.cpu.tle_ops import sdot_gemv_prequant

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    pack_weights_sdot,
    pack_weights_sdot_blocked,
)


@triton.jit
def _quantize_f32_w8_kernel(
    x_ptr,
    x_q_ptr,
    x_scale_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ordinary Triton: FP32 activation -> shared INT8 vector and scale."""
    absmax = tl.zeros((1,), dtype=tl.float32)
    for off in range(0, K, BLOCK_K):
        cols = off + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + cols, mask=cols < K, other=0.0)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x), axis=0))
    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    tl.store(x_scale_ptr + tl.arange(0, 1), scale)
    for off in range(0, K, BLOCK_K):
        cols = off + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + cols, mask=cols < K, other=0.0)
        scaled = tl.minimum(tl.maximum(x / scale, -128.0), 127.0)
        q = libdevice.rint(scaled).to(tl.int8)
        tl.store(x_q_ptr + cols, q, mask=cols < K)


@triton.jit
def _f32_w8_prequant_gemv_kernel(
    x_q_ptr,
    x_scale_ptr,
    packed_ptr,
    weight_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compiler-visible packed SDOT range; no external GEMV call."""
    n_start = tl.program_id(0) * BLOCK_N
    sdot_gemv_prequant(
        x_q_ptr,
        x_scale_ptr,
        packed_ptr,
        weight_scale_ptr,
        out_ptr,
        K,
        N,
        n_start,
        BLOCK_N,
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


def last_compiled(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    return list(cache.values())[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--block-n", type=int, default=512)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--runtime-so",
        default="/home/kevin/triton-opt-cpu/python/triton/_C/"
        "libTritonCPURuntime.so",
    )
    args = parser.parse_args()
    if args.k % 4 or args.n % args.block_n or args.block_n % 4:
        raise ValueError("invalid K/N/BLOCK_N divisibility")

    torch.manual_seed(47)
    x = torch.randn(args.k, dtype=torch.float32)
    weight_kn = torch.randint(-127, 128, (args.k, args.n), dtype=torch.int8)
    scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    packed = pack_weights_sdot_blocked(
        pack_weights_sdot(weight_kn), args.block_n
    )
    x_q = torch.empty(args.k, dtype=torch.int8)
    x_scale = torch.empty(1, dtype=torch.float32)
    codegen_out = torch.empty(args.n, dtype=torch.float32)
    runtime_out = torch.empty_like(codegen_out)
    runtime_x_q = torch.empty_like(x_q)

    def run_codegen() -> None:
        _quantize_f32_w8_kernel[(1,)](
            x,
            x_q,
            x_scale,
            K=args.k,
            BLOCK_K=args.block_k,
        )
        _f32_w8_prequant_gemv_kernel[(args.n // args.block_n,)](
            x_q,
            x_scale,
            packed,
            scale,
            codegen_out,
            K=args.k,
            N=args.n,
            BLOCK_N=args.block_n,
        )

    runtime = ctypes.CDLL(args.runtime_so)
    quant = runtime.sdot_quant_act_f32
    quant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    quant.restype = ctypes.c_float
    gemv = runtime.sdot_gemv_blk_prequant_f32_range
    gemv.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ] + [ctypes.c_int64] * 6
    gemv.restype = None

    def run_runtime() -> None:
        activation_scale = quant(x.data_ptr(), runtime_x_q.data_ptr(), args.k)
        gemv(
            runtime_x_q.data_ptr(),
            activation_scale,
            packed.data_ptr(),
            scale.data_ptr(),
            runtime_out.data_ptr(),
            args.k,
            args.n,
            args.n // 4,
            args.block_n // 4,
            0,
            args.n // args.block_n,
        )

    run_codegen()
    run_runtime()
    if not torch.equal(x_q, runtime_x_q) or not torch.equal(
        codegen_out, runtime_out
    ):
        delta = (codegen_out - runtime_out).abs()
        raise AssertionError(
            f"mismatch xq={torch.count_nonzero(x_q != runtime_x_q).item()} "
            f"out={torch.count_nonzero(delta).item()} max={delta.max().item()}"
        )

    codegen_us = median_us(
        run_codegen, args.warmup, args.iters, args.batches
    )
    runtime_us = median_us(
        run_runtime, args.warmup, args.iters, args.batches
    )
    quant_compiled = last_compiled(_quantize_f32_w8_kernel)
    gemv_compiled = last_compiled(_f32_w8_prequant_gemv_kernel)
    quant_llir = quant_compiled.asm["llir"]
    gemv_llir = gemv_compiled.asm["llir"]
    gemv_asm = gemv_compiled.asm["asm"].lower()
    print(
        f"PASS K={args.k} N={args.n} BLOCK_N={args.block_n}\n"
        f"triton_split_python_us={codegen_us:.3f}\n"
        f"ggml_c_runtime_us={runtime_us:.3f}\n"
        f"triton_over_c={codegen_us / runtime_us:.3f}x\n"
        f"bit_exact={torch.equal(codegen_out, runtime_out)}\n"
        f"quant_external_runtime={quant_llir.count('sdot_quant_act_f32')}\n"
        f"gemv_external_runtime="
        f"{gemv_llir.count('sdot_gemv_blk_prequant_f32_range')}\n"
        f"gemv_llir_sdot={gemv_llir.count('llvm.aarch64.neon.sdot')}\n"
        f"gemv_asm_sdot={gemv_asm.count('sdot')}"
    )


if __name__ == "__main__":
    main()
