#!/usr/bin/env python3
"""M=1 INT8 GEMV: ordinary Triton LLVM codegen versus packed C runtime.

Run this script in a fresh process when changing
TRITON_CPU_DISABLE_SVE2_I8MM; the flag is consumed while compiling the
kernel.  Both measurements include their normal Python-to-native launch
path, while the reported codegen evidence comes from the compiled Triton
artifact itself.
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
def m1_int8_codegen_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((1, BLOCK_N), dtype=tl.int32)
    for off in range(0, K, BLOCK_K):
        offs_k = off + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_n, tl.reshape(acc, (BLOCK_N,)))


def pack_sdot(weight_kn: torch.Tensor) -> torch.Tensor:
    k, n = weight_kn.shape
    return (
        weight_kn.reshape(k // 4, 4, n // 4, 4)
        .permute(0, 2, 3, 1)
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
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--bk", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--runtime-so",
        default="/home/kevin/triton-opt-cpu/python/triton/_C/"
        "libTritonCPURuntime.so",
    )
    args = parser.parse_args()

    if args.k % 4 or args.n % args.bn or args.bn % 4 or args.bk % 4:
        raise ValueError("K%4, N%BLOCK_N, BLOCK_N%4 and BLOCK_K%4 must be zero")

    torch.manual_seed(23)
    activation = torch.randint(-128, 128, (args.k,), dtype=torch.int8)
    weight = torch.randint(-128, 128, (args.k, args.n), dtype=torch.int8)
    packed = pack_sdot(weight)
    codegen_out = torch.empty(args.n, dtype=torch.int32)
    runtime_out = torch.empty_like(codegen_out)

    def run_codegen() -> None:
        m1_int8_codegen_kernel[(args.n // args.bn,)](
            activation,
            weight,
            codegen_out,
            K=args.k,
            N=args.n,
            BLOCK_N=args.bn,
            BLOCK_K=args.bk,
        )

    runtime = ctypes.CDLL(args.runtime_so)
    runtime_gemv = runtime.sdot_gemv_m1_prepacked
    runtime_gemv.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3
    runtime_gemv.restype = None

    def run_runtime() -> None:
        runtime_gemv(
            activation.data_ptr(),
            packed.data_ptr(),
            runtime_out.data_ptr(),
            args.k,
            args.n,
            args.n // 4,
        )

    run_codegen()
    run_runtime()
    reference = activation.to(torch.int32) @ weight.to(torch.int32)
    if not torch.equal(codegen_out, reference):
        raise AssertionError("ordinary Triton codegen result mismatch")
    if not torch.equal(runtime_out, reference):
        raise AssertionError("packed C runtime result mismatch")

    codegen_us = median_us(
        run_codegen, args.warmup, args.iters, args.batches
    )
    runtime_us = median_us(
        run_runtime, args.warmup, args.iters, args.batches
    )

    device_cache = next(iter(m1_int8_codegen_kernel.device_caches.values()))[0]
    compiled = list(device_cache.values())[-1]
    llir = compiled.asm["llir"]
    assembly = compiled.asm["asm"].lower()
    print(
        f"PASS M=1 N={args.n} K={args.k} BN={args.bn} BK={args.bk}\n"
        f"ordinary_triton_us={codegen_us:.3f}\n"
        f"packed_c_runtime_us={runtime_us:.3f}\n"
        f"triton_over_runtime={codegen_us / runtime_us:.3f}x\n"
        f"llir_neon_sdot={llir.count('llvm.aarch64.neon.sdot')}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"llir_external_sdot_gemv={llir.count('sdot_gemv')}"
    )


if __name__ == "__main__":
    main()
