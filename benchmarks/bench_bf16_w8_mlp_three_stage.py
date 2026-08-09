#!/usr/bin/env python3
"""Register-light ordinary-dot BF16-W8 SwiGLU experiment."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_gems.runtime.backend._arm.ops.silu_and_mul import (
    _sleef_expf_u10_inline,
)

from bench_bf16_w8a8_ordinary_split import (
    _quantize_bf16_w8_kernel,
    last_compiled,
    median_us,
)
from bench_bf16_w8a8_wide_split import (
    _w8a8_wide_gemv_kernel,
    pack_w8_wide,
)


@triton.jit
def _bf16_swiglu_kernel(
    gate_up_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        gate = tl.load(gate_up_ptr + cols).to(tl.float32)
        up = tl.load(gate_up_ptr + N + cols).to(tl.float32)
        silu = gate / (1.0 + tl.exp(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + cols, (silu * up).to(tl.bfloat16))


@triton.jit
def _bf16_swiglu_inline_exp_kernel(
    gate_up_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        gate = tl.load(gate_up_ptr + cols).to(tl.float32)
        up = tl.load(gate_up_ptr + N + cols).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + cols, (silu * up).to(tl.bfloat16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--activation-block", type=int, default=16)
    parser.add_argument("--unroll", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(866)
    x = torch.randn(args.k, dtype=torch.bfloat16) * 0.2
    gate_weight = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    up_weight = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8
    )
    joined_weight = torch.cat((gate_weight, up_weight), dim=1)
    packed = pack_w8_wide(joined_weight, args.block_n)
    gate_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    up_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    joined_scale = torch.cat((gate_scale, up_scale)).contiguous()
    x_q = torch.empty(args.k, dtype=torch.int8)
    x_scale = torch.empty(1, dtype=torch.float32)
    gate_up = torch.empty(args.n * 2, dtype=torch.bfloat16)
    output = torch.empty(args.n, dtype=torch.bfloat16)
    inline_output = torch.empty_like(output)

    def run_quant() -> None:
        _quantize_bf16_w8_kernel[(1,)](
            x, x_q, x_scale, K=args.k, BLOCK_K=16
        )

    def run_gemv() -> None:
        _w8a8_wide_gemv_kernel[(1,)](
            x_q,
            x_scale,
            packed,
            joined_scale,
            gate_up,
            0,
            args.n * 2 // args.block_n,
            K=args.k,
            N=args.n * 2,
            BLOCK_N=args.block_n,
            UNROLL=args.unroll,
        )

    def run_activation() -> None:
        _bf16_swiglu_kernel[(1,)](
            gate_up,
            output,
            N=args.n,
            BLOCK_N=args.activation_block,
        )

    def run_inline_activation() -> None:
        _bf16_swiglu_inline_exp_kernel[(1,)](
            gate_up,
            inline_output,
            N=args.n,
            BLOCK_N=args.activation_block,
        )

    def run_all() -> None:
        run_quant()
        run_gemv()
        run_activation()

    run_all()
    run_inline_activation()
    xf = x.float()
    absmax = xf.abs().max().clamp(min=1.0e-8)
    expected_scale = absmax / 127.0
    expected_q = (
        (xf * (127.0 / absmax)).clamp(-128, 127).round().to(torch.int8)
    )
    gate = (
        (expected_q.to(torch.int32) @ gate_weight.to(torch.int32)).float()
        * expected_scale
        * gate_scale
    ).to(torch.bfloat16)
    up = (
        (expected_q.to(torch.int32) @ up_weight.to(torch.int32)).float()
        * expected_scale
        * up_scale
    ).to(torch.bfloat16)
    expected = (F.silu(gate) * up).to(torch.bfloat16)
    assert torch.equal(x_q, expected_q)
    assert torch.equal(output, expected)
    inline_exact = torch.equal(inline_output, expected)

    total_us = median_us(run_all, args.warmup, args.iters, args.batches)
    gemv_us = median_us(run_gemv, args.warmup, args.iters, args.batches)
    activation_us = median_us(
        run_activation, args.warmup, args.iters, args.batches
    )
    inline_activation_us = median_us(
        run_inline_activation, args.warmup, args.iters, args.batches
    )
    gemv_compiled = last_compiled(_w8a8_wide_gemv_kernel)
    activation_compiled = last_compiled(_bf16_swiglu_kernel)
    inline_compiled = last_compiled(_bf16_swiglu_inline_exp_kernel)
    gemv_asm = gemv_compiled.asm["asm"].lower()
    activation_asm = activation_compiled.asm["asm"].lower()
    inline_asm = inline_compiled.asm["asm"].lower()
    inline_llir = inline_compiled.asm["llir"].lower()
    print(
        f"PASS three-stage BF16-W8 SwiGLU K={args.k} N={args.n} "
        f"BN={args.block_n}\n"
        f"three_python_launches_us={total_us:.3f}\n"
        f"gemv_python_us={gemv_us:.3f}\n"
        f"activation_python_us={activation_us:.3f}\n"
        f"inline_activation_python_us={inline_activation_us:.3f}\n"
        f"inline_bit_exact={inline_exact}\n"
        f"bit_exact={torch.equal(output, expected)}\n"
        f"gemv_stack_load_store={sum('[sp' in line and line.lstrip().startswith(('ld', 'st')) for line in gemv_asm.splitlines())}\n"
        f"activation_stack_load_store={sum('[sp' in line and line.lstrip().startswith(('ld', 'st')) for line in activation_asm.splitlines())}\n"
        f"gemv_asm_sdot={gemv_asm.count('sdot')}\n"
        f"activation_asm_lines={len(activation_asm.splitlines())}\n"
        f"inline_activation_asm_lines={len(inline_asm.splitlines())}\n"
        f"inline_stack_load_store={sum('[sp' in line and line.lstrip().startswith(('ld', 'st')) for line in inline_asm.splitlines())}\n"
        f"inline_external_calls={sum(' call ' in line and 'llvm.' not in line for line in inline_llir.splitlines())}"
    )


if __name__ == "__main__":
    main()
