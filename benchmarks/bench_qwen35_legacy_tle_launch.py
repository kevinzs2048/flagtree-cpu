#!/usr/bin/env python3
"""Measure the legacy Qwen3.5 TLE launch path, including Triton launch."""

from __future__ import annotations

import statistics
import time

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import (
    causal_conv1d_update,
    flash_attn_decode,
    gated_delta_decode,
    rms_norm,
    rms_norm_gated,
)


@triton.jit
def rms_kernel(x, weight, out, D: tl.constexpr, eps: tl.constexpr):
    rms_norm(x, weight, out, D, eps)


@triton.jit
def gated_kernel(
    x,
    gate,
    weight,
    out,
    M: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    rms_norm_gated(x, gate, weight, out, M, D, eps)


@triton.jit
def conv_kernel(
    hidden,
    state,
    weight,
    bias,
    out,
    B: tl.constexpr,
    C: tl.constexpr,
):
    causal_conv1d_update(hidden, state, weight, bias, out, B, C, 4, 1, 0)


@triton.jit
def delta_kernel(q, k, v, g, beta, state, out):
    gated_delta_decode(q, k, v, g, beta, state, out, 1, 16, 128, 128, 1)


@triton.jit
def attention_kernel(q, k, v, out, seq_len: tl.constexpr):
    flash_attn_decode(
        q,
        k,
        v,
        out,
        seq_len,
        128,
        128**-0.5,
        16,
        8,
        128,
        128,
    )


def median_us(function, iterations=1000, batches=7):
    for _ in range(10):
        function()
    samples = []
    for _ in range(batches):
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - begin) / iterations / 1000.0)
    return statistics.median(samples)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(3535)
    eps = 1.0e-6

    x = torch.randn(2560, dtype=torch.bfloat16)
    weight = torch.randn(2560, dtype=torch.bfloat16)
    out = torch.empty_like(x)

    gated_x = torch.randn(16, 128, dtype=torch.bfloat16)
    gate = torch.randn_like(gated_x)
    gated_weight = torch.randn(128, dtype=torch.bfloat16)
    gated_out = torch.empty_like(gated_x)

    hidden = torch.randn(8192, dtype=torch.bfloat16)
    state = torch.randn(1, 8192, 4, dtype=torch.bfloat16)
    conv_weight = torch.randn(8192, 4, dtype=torch.bfloat16)
    dummy_bias = torch.zeros(1, dtype=torch.bfloat16)
    conv_out = torch.empty_like(hidden)

    q = torch.randn(1, 16, 128, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, 16, dtype=torch.float32) * 0.2
    beta = torch.sigmoid(torch.randn(1, 16, dtype=torch.float32))
    delta_state = torch.randn(1, 16, 128, 128, dtype=torch.float32) * 0.01
    delta_out = torch.empty_like(v)
    attn_q = torch.randn(16, 128, dtype=torch.bfloat16)
    attn_k = torch.randn(8, 128, 128, dtype=torch.bfloat16)
    attn_v = torch.randn_like(attn_k)
    attn_out = torch.empty_like(attn_q)

    def run_rms():
        rms_kernel[(1,)](x, weight, out, D=2560, eps=eps)

    def run_gated():
        gated_kernel[(1,)](
            gated_x,
            gate,
            gated_weight,
            gated_out,
            M=16,
            D=128,
            eps=eps,
        )

    def run_conv():
        conv_kernel[(1,)](
            hidden,
            state,
            conv_weight,
            dummy_bias,
            conv_out,
            B=1,
            C=8192,
        )

    def run_delta():
        delta_kernel[(1,)](q, k, v, g, beta, delta_state, delta_out)

    def run_attention():
        attention_kernel[(1,)](attn_q, attn_k, attn_v, attn_out, 128)

    print(
        f"legacy_tle_rms_us={median_us(run_rms):.3f}\n"
        f"legacy_tle_gated_us={median_us(run_gated):.3f}\n"
        f"legacy_tle_conv_us={median_us(run_conv):.3f}\n"
        f"legacy_tle_delta_us={median_us(run_delta, 100, 7):.3f}\n"
        f"legacy_tle_attention_us={median_us(run_attention, 100, 7):.3f}"
    )


if __name__ == "__main__":
    main()
