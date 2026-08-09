#!/usr/bin/env python3
"""Compare compiler-visible M=1 attention with ATen and the legacy C runtime."""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import time

import torch

sys.meta_path = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ != "_flag_gems_editable"
]

from flag_gems.runtime.backend._arm.ops.attention import (
    _aten_sdpa,
    _flash_attn_decode_codegen_kernel,
    _flash_attn_decode_pv_codegen_kernel,
    _flash_attn_decode_scores_codegen_kernel,
    scaled_dot_product_attention,
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


def error(reference: torch.Tensor, output: torch.Tensor) -> tuple[float, float]:
    delta = reference.float() - output.float()
    return float(delta.norm() / reference.float().norm()), float(
        delta.abs().max()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--tle-runtime",
        default="/home/cix/triton-cpu-int8/triton/_C/libTritonCPURuntime.so",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    query = torch.randn(
        1, args.q_heads, 1, args.head_dim, dtype=torch.bfloat16
    )
    key = torch.randn(
        1,
        args.kv_heads,
        args.seq_len,
        args.head_dim,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    runtime_out = torch.empty_like(query)
    runtime = ctypes.CDLL(args.tle_runtime)
    runtime_function = runtime.flash_attn_decode_bf16
    runtime_function.argtypes = [ctypes.c_void_p] * 4 + [
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_float,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_long,
    ]
    runtime_function.restype = None

    def run_aten():
        return _aten_sdpa(query, key, value, enable_gqa=True)

    def run(implementation: str):
        os.environ["FLAGGEMS_ARM_ATTN_DECODE_IMPL"] = implementation
        return scaled_dot_product_attention(
            query, key, value, enable_gqa=True
        )

    def run_runtime():
        runtime_function(
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            runtime_out.data_ptr(),
            args.seq_len,
            args.head_dim,
            args.head_dim**-0.5,
            args.q_heads,
            args.kv_heads,
            key.stride(2),
            value.stride(2),
        )
        return runtime_out

    codegen_out = run("codegen")
    staged_out = run("staged")
    auto_out = run("auto")
    runtime_router_out = run("runtime")
    c_runtime_out = run_runtime()
    reference = run_aten()
    codegen_error = error(reference, codegen_out)
    staged_error = error(reference, staged_out)
    auto_error = error(reference, auto_out)
    staged_codegen_error = error(codegen_out, staged_out)
    runtime_router_error = error(reference, runtime_router_out)
    runtime_error = error(reference, c_runtime_out)

    # Keep allocation, Python routing and tensor-view construction out of this
    # closure.  This is the fair counterpart to ``run_runtime`` above: both
    # time only an already-compiled compute kernel with preallocated buffers.
    q_flat = query.squeeze(0).squeeze(1).contiguous()
    k_flat = key.squeeze(0).contiguous()
    v_flat = value.squeeze(0).contiguous()
    launch_out = torch.empty(args.q_heads, args.head_dim, dtype=torch.bfloat16)
    block_n = int(os.getenv("FLAGGEMS_ARM_ATTN_CODEGEN_BLOCK_N", "1"))
    score_block = int(
        os.getenv("FLAGGEMS_ARM_ATTN_STAGED_SCORE_BLOCK", "16")
    )
    block_d = int(os.getenv("FLAGGEMS_ARM_ATTN_STAGED_BLOCK_D", "64"))
    scores = torch.empty(
        args.q_heads * args.seq_len + args.q_heads, dtype=torch.float32
    )
    staged_out_flat = torch.empty(
        args.q_heads, args.head_dim, dtype=torch.bfloat16
    )

    def run_codegen_launch():
        _flash_attn_decode_codegen_kernel[(args.q_heads,)](
            q_flat,
            k_flat,
            v_flat,
            launch_out,
            args.seq_len,
            args.head_dim**-0.5,
            args.q_heads,
            args.kv_heads,
            HEAD_DIM=args.head_dim,
            BLOCK_N=block_n,
        )

    def run_runtime_router():
        run("runtime")

    def run_auto_router():
        run("auto")

    def run_staged_launch():
        _flash_attn_decode_scores_codegen_kernel[(args.q_heads,)](
            q_flat,
            k_flat,
            scores,
            args.seq_len,
            args.head_dim**-0.5,
            args.q_heads,
            args.kv_heads,
            HEAD_DIM=args.head_dim,
            SCORE_BLOCK=score_block,
        )
        _flash_attn_decode_pv_codegen_kernel[
            (args.q_heads * (args.head_dim // block_d),)
        ](
            v_flat,
            scores,
            staged_out_flat,
            args.seq_len,
            args.q_heads,
            args.kv_heads,
            HEAD_DIM=args.head_dim,
            BLOCK_D=block_d,
        )

    def run_staged_router():
        run("staged")

    def run_codegen():
        run("codegen")

    codegen_us = median_us(
        run_codegen, args.warmup, args.iters, args.batches
    )
    codegen_launch_us = median_us(
        run_codegen_launch, args.warmup, args.iters, args.batches
    )
    runtime_router_us = median_us(
        run_runtime_router, args.warmup, args.iters, args.batches
    )
    auto_router_us = median_us(
        run_auto_router, args.warmup, args.iters, args.batches
    )
    staged_launch_us = median_us(
        run_staged_launch, args.warmup, args.iters, args.batches
    )
    staged_router_us = median_us(
        run_staged_router, args.warmup, args.iters, args.batches
    )
    runtime_us = median_us(
        run_runtime, args.warmup, args.iters, args.batches
    )
    aten_us = median_us(run_aten, args.warmup, args.iters, args.batches)

    cache = next(
        iter(_flash_attn_decode_codegen_kernel.device_caches.values())
    )[0]
    compiled = list(cache.values())[-1]
    llir = compiled.asm["llir"]
    assembly = compiled.asm["asm"].lower()
    print(
        f"PASS N={args.seq_len} Hq={args.q_heads} Hkv={args.kv_heads} "
        f"D={args.head_dim} seed={args.seed}\n"
        f"triton_codegen_router_us={codegen_us:.3f}\n"
        f"triton_codegen_launch_us={codegen_launch_us:.3f}\n"
        f"triton_staged_router_us={staged_router_us:.3f}\n"
        f"triton_staged_launch_us={staged_launch_us:.3f}\n"
        f"triton_auto_router_us={auto_router_us:.3f}\n"
        f"runtime_router_us={runtime_router_us:.3f}\n"
        f"legacy_c_runtime_us={runtime_us:.3f}\n"
        f"aten_flash_us={aten_us:.3f}\n"
        f"codegen_router_over_runtime_router="
        f"{codegen_us / runtime_router_us:.3f}x\n"
        f"codegen_launch_over_c={codegen_launch_us / runtime_us:.3f}x\n"
        f"staged_launch_over_c={staged_launch_us / runtime_us:.3f}x\n"
        f"codegen_over_aten={codegen_us / aten_us:.3f}x\n"
        f"codegen_relative_l2={codegen_error[0]:.8f}\n"
        f"codegen_max_abs={codegen_error[1]:.8f}\n"
        f"staged_relative_l2={staged_error[0]:.8f}\n"
        f"staged_max_abs={staged_error[1]:.8f}\n"
        f"auto_relative_l2={auto_error[0]:.8f}\n"
        f"auto_max_abs={auto_error[1]:.8f}\n"
        f"staged_vs_codegen_relative_l2={staged_codegen_error[0]:.8f}\n"
        f"staged_vs_codegen_max_abs={staged_codegen_error[1]:.8f}\n"
        f"runtime_relative_l2={runtime_error[0]:.8f}\n"
        f"runtime_max_abs={runtime_error[1]:.8f}\n"
        f"runtime_router_relative_l2={runtime_router_error[0]:.8f}\n"
        f"runtime_router_max_abs={runtime_router_error[1]:.8f}\n"
        f"llir_external_runtime={llir.count('flash_attn_decode_bf16')}\n"
        f"llir_exp={llir.count('Sleef_exp') + llir.count('llvm.exp')}\n"
        f"asm_fmla={assembly.count('fmla')}"
    )


if __name__ == "__main__":
    main()
