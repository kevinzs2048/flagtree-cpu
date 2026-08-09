#!/usr/bin/env python3
"""ggml-style FP32 activation/output W8 decode through compiler SDOT lowering."""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import (
    sdot_gemv_fused_bf16 as _cpu_fused_gemv,
)

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    pack_weights_sdot,
    pack_weights_sdot_blocked,
)


@triton.jit
def _f32_w8_gemv_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n_start = tl.program_id(0) * BLOCK_N
    # The compiler op is type-polymorphic despite its compatibility name:
    # FP32 pointer types select FP32 activation loads and FP32 stores.
    _cpu_fused_gemv(
        x_ptr,
        packed_ptr,
        scale_ptr,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--block-n", type=int, default=512)
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

    torch.manual_seed(41)
    x = torch.randn(args.k, dtype=torch.float32)
    weight_kn = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    packed_kmajor = pack_weights_sdot(weight_kn)
    packed = pack_weights_sdot_blocked(packed_kmajor, args.block_n)
    codegen_out = torch.empty(args.n, dtype=torch.float32)
    runtime_out = torch.empty_like(codegen_out)
    xq = torch.empty(args.k, dtype=torch.int8)

    def run_codegen() -> None:
        _f32_w8_gemv_kernel[(args.n // args.block_n,)](
            x,
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
        x_scale = quant(x.data_ptr(), xq.data_ptr(), args.k)
        gemv(
            xq.data_ptr(),
            x_scale,
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
    if not torch.equal(codegen_out, runtime_out):
        delta = (codegen_out - runtime_out).abs()
        raise AssertionError(
            f"mismatch count={torch.count_nonzero(delta).item()} "
            f"max={delta.max().item()}"
        )
    codegen_us = median_us(
        run_codegen, args.warmup, args.iters, args.batches
    )
    runtime_us = median_us(
        run_runtime, args.warmup, args.iters, args.batches
    )
    cache = next(iter(_f32_w8_gemv_kernel.device_caches.values()))[0]
    compiled = list(cache.values())[-1]
    llir = compiled.asm["llir"]
    assembly = compiled.asm["asm"].lower()
    print(
        f"PASS K={args.k} N={args.n} BLOCK_N={args.block_n}\n"
        f"triton_codegen_us={codegen_us:.3f}\n"
        f"ggml_c_runtime_us={runtime_us:.3f}\n"
        f"triton_over_c={codegen_us / runtime_us:.3f}x\n"
        f"bit_exact={torch.equal(codegen_out, runtime_out)}\n"
        f"llir_external_runtime="
        f"{llir.count('sdot_gemv_blk_prequant_f32_range')}\n"
        f"llir_sdot={llir.count('llvm.aarch64.neon.sdot')}\n"
        f"asm_sdot={assembly.count('sdot')}"
    )


if __name__ == "__main__":
    main()
