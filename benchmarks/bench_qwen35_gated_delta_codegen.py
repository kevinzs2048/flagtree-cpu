#!/usr/bin/env python3
"""Microbenchmark ordinary-Triton Qwen3.5 recurrent gated-delta decode."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import statistics
import sys
import time

import torch


sys.meta_path = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ != "_flag_gems_editable"
]


def median_us(function, iterations: int, batches: int) -> float:
    for _ in range(5):
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
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--k-dim", type=int, default=128)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--block-v", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-l2", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--tle-runtime",
        default="/home/cix/triton-cpu-int8/triton/_C/libTritonCPURuntime.so",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(35128)

    module = importlib.import_module(
        "flag_gems.runtime.backend._arm.fused.patch_qwen3_5_gated_delta"
    )
    h, kd, vd = args.heads, args.k_dim, args.v_dim
    query = torch.randn(1, h, kd, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(1, h, vd, dtype=torch.bfloat16)
    g = -torch.rand(1, h, dtype=torch.float32) * 0.2
    beta = torch.sigmoid(torch.randn(1, h, dtype=torch.float32))
    initial_state = torch.randn(1, h, kd, vd, dtype=torch.float32) * 0.01
    reference_state = initial_state.clone()
    ordinary_state = initial_state.clone()
    tle_state = initial_state.clone()
    ordinary_out = torch.empty(1, h, vd, dtype=torch.bfloat16)
    tle_out = torch.empty(1, h, vd, dtype=torch.float32)

    def reference():
        nonlocal reference_state
        q = query.float()
        k = key.float()
        v = value.float()
        if not args.no_l2:
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1.0e-6)
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1.0e-6)
        q = q * (1.0 / (kd**0.5))
        state = reference_state * g.exp().unsqueeze(-1).unsqueeze(-1)
        memory = (state * k.unsqueeze(-1)).sum(-2)
        delta = (v - memory) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        result = (state * q.unsqueeze(-1)).sum(-2)
        reference_state = state
        return result.to(torch.bfloat16)

    grid = (h * ((vd + args.block_v - 1) // args.block_v),)

    def ordinary():
        module._gated_delta_decode_kernel[grid](
            query,
            key,
            value,
            g,
            beta,
            ordinary_state,
            ordinary_out,
            B=1,
            H=h,
            k_dim=kd,
            v_dim=vd,
            use_l2norm=0 if args.no_l2 else 1,
            BLOCK_K=args.block_k,
            BLOCK_V=args.block_v,
            num_warps=1,
            num_stages=1,
        )
        return ordinary_out

    runtime = ctypes.CDLL(args.tle_runtime)
    tle_function = runtime.standalone_gated_delta_decode_fp32
    tle_function.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_long] * 5
    tle_function.restype = None
    q_float = query.float().contiguous()
    k_float = key.float().contiguous()
    value_float = value.float().contiguous()

    def tle_runtime():
        tle_function(
            q_float.data_ptr(),
            k_float.data_ptr(),
            value_float.data_ptr(),
            g.data_ptr(),
            beta.data_ptr(),
            tle_state.data_ptr(),
            tle_out.data_ptr(),
            1,
            h,
            kd,
            vd,
            0 if args.no_l2 else 1,
        )
        return tle_out.to(torch.bfloat16)

    expected = reference()
    actual = ordinary()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        ordinary_state, reference_state, rtol=2e-4, atol=2e-5
    )
    ordinary_error = (actual.float() - expected.float()).abs().max().item()
    tle_actual = tle_runtime()
    torch.testing.assert_close(tle_actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(tle_state, reference_state, rtol=2e-4, atol=2e-5)
    tle_error = (tle_actual.float() - expected.float()).abs().max().item()

    reference_us = median_us(reference, args.iterations, args.batches)
    ordinary_us = median_us(ordinary, args.iterations, args.batches)
    tle_us = median_us(tle_runtime, args.iterations, args.batches)
    compiled = last_compiled(module._gated_delta_decode_kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    stack = sum(
        "[sp" in line and line.lstrip().startswith(("ld", "st"))
        for line in assembly.splitlines()
    )
    calls = sum(
        " call " in line and "llvm." not in line for line in llir.splitlines()
    )
    print(
        "PASS Qwen3.5 gated-delta ordinary codegen\n"
        f"shape=1x{h}x{kd}x{vd} threads={args.threads} "
        f"reference_us={reference_us:.3f} triton_us={ordinary_us:.3f} "
        f"triton_over_reference={ordinary_us / reference_us:.3f}x\n"
        f"tle_runtime_us={tle_us:.3f} triton_over_tle={ordinary_us / tle_us:.3f}x\n"
        f"max_abs_error={ordinary_error:.6f} tle_max_abs_error={tle_error:.6f} "
        f"asm_lines={len(assembly.splitlines())} stack_refs={stack} calls={calls}"
    )


if __name__ == "__main__":
    main()
