#!/usr/bin/env python3
"""Shape-specialized ordinary-Triton fused add + BF16 RMSNorm AOT."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from bench_bf16_w8a8_ordinary_split import last_compiled, median_us


@triton.jit
def _fused_add_rms_aot_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        input_value = tl.load(input_ptr + row * N + cols).to(tl.float32)
        residual = tl.load(residual_ptr + row * N + cols).to(tl.float32)
        updated = (input_value + residual).to(tl.bfloat16)
        tl.store(residual_ptr + row * N + cols, updated)
        updated_fp32 = updated.to(tl.float32)
        sum_sq += tl.sum(updated_fp32 * updated_fp32, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / N + EPS)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        updated = tl.load(residual_ptr + row * N + cols).to(tl.float32)
        weight = tl.load(weight_ptr + cols)
        normalized = (updated * rrms).to(tl.bfloat16)
        tl.store(
            input_ptr + row * N + cols,
            (normalized * weight).to(tl.bfloat16),
        )


@triton.jit
def _vllm_fused_add_rms_aot_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Match current vLLM CPU forward_static residual semantics.

    vLLM normalizes the unrounded FP32 sum but separately materializes a BF16
    residual. Delay both stores until the second pass so that the original
    input and residual can be reloaded to reproduce that sum without a temp.
    """
    row = tl.program_id(0)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        input_value = tl.load(input_ptr + row * N + cols).to(tl.float32)
        residual = tl.load(residual_ptr + row * N + cols).to(tl.float32)
        updated_fp32 = input_value + residual
        sum_sq += tl.sum(updated_fp32 * updated_fp32, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / N + EPS)
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        input_value = tl.load(input_ptr + row * N + cols).to(tl.float32)
        residual = tl.load(residual_ptr + row * N + cols).to(tl.float32)
        updated_fp32 = input_value + residual
        weight = tl.load(weight_ptr + cols)
        normalized = (updated_fp32 * rrms).to(tl.bfloat16)
        tl.store(
            input_ptr + row * N + cols,
            (normalized * weight).to(tl.bfloat16),
        )
        tl.store(
            residual_ptr + row * N + cols,
            updated_fp32.to(tl.bfloat16),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--vllm-semantics", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(868)
    input_value = torch.randn(args.m, args.n, dtype=torch.bfloat16)
    residual = torch.randn_like(input_value)
    weight = torch.randn(args.n, dtype=torch.bfloat16)
    input_initial = input_value.clone()
    residual_initial = residual.clone()

    def reset() -> None:
        input_value.copy_(input_initial)
        residual.copy_(residual_initial)

    kernel = (
        _vllm_fused_add_rms_aot_kernel
        if args.vllm_semantics
        else _fused_add_rms_aot_kernel
    )

    def run() -> None:
        kernel[(args.m,)](
            input_value,
            residual,
            weight,
            M=args.m,
            N=args.n,
            EPS=args.eps,
            BLOCK_N=args.block_n,
        )

    reset()
    run()
    expected_residual = (input_initial + residual_initial).to(torch.bfloat16)
    updated_fp32 = (
        input_initial.float() + residual_initial.float()
        if args.vllm_semantics
        else expected_residual.float()
    )
    rrms = torch.rsqrt(
        (updated_fp32 * updated_fp32).sum(dim=-1, keepdim=True) / args.n
        + args.eps
    )
    expected_input = (
        (updated_fp32 * rrms).to(torch.bfloat16) * weight
    ).to(torch.bfloat16)
    assert torch.equal(residual, expected_residual)
    assert torch.equal(input_value, expected_input)

    # Repeated execution intentionally measures only the kernel. Values evolve
    # in place exactly as they do between model layers.
    latency = median_us(run, args.warmup, args.iters, args.batches)
    compiled = last_compiled(kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    print(
        f"PASS ordinary fused-add RMSNorm M={args.m} N={args.n} "
        f"vllm_semantics={args.vllm_semantics}\n"
        f"python_launch_us={latency:.3f}\n"
        "bit_exact=True\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"stack_load_store={sum('[sp' in line and line.lstrip().startswith(('ld', 'st')) for line in assembly.splitlines())}\n"
        f"external_calls={sum(' call ' in line and 'llvm.' not in line for line in llir.splitlines())}"
    )


if __name__ == "__main__":
    main()
