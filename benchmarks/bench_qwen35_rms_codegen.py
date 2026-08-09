#!/usr/bin/env python3
"""Microbenchmark ordinary-Triton Qwen3.5 RMSNorm variants."""

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
    for _ in range(10):
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


def audit(kernel) -> tuple[int, int, int]:
    compiled = last_compiled(kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    stack = sum(
        "[sp" in line and line.lstrip().startswith(("ld", "st"))
        for line in assembly.splitlines()
    )
    calls = sum(" call " in line and "llvm." not in line for line in llir.splitlines())
    return len(assembly.splitlines()), stack, calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=2560)
    parser.add_argument("--gated-m", type=int, default=16)
    parser.add_argument("--gated-d", type=int, default=128)
    parser.add_argument("--rms-block", type=int, default=16)
    parser.add_argument("--gated-block", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--tle-runtime",
        default="/home/cix/triton-cpu-int8/triton/_C/libTritonCPURuntime.so",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(3535)

    rms_module = importlib.import_module(
        "flag_gems.runtime.backend._arm.fused.patch_qwen3_5_rmsnorm"
    )
    gated_module = importlib.import_module(
        "flag_gems.runtime.backend._arm.fused.patch_qwen3_5_rmsnorm_gated"
    )

    x = torch.randn(1, args.d, dtype=torch.bfloat16)
    weight = torch.randn(args.d, dtype=torch.float32) * 0.1
    out = torch.empty_like(x)
    tle_out = torch.empty_like(x)
    eps = 1.0e-6

    runtime = ctypes.CDLL(args.tle_runtime)
    tle_rms_function = runtime.standalone_rms_norm_bf16
    tle_rms_function.argtypes = [ctypes.c_void_p] * 3 + [
        ctypes.c_long,
        ctypes.c_float,
    ]
    tle_rms_function.restype = None
    # The legacy runtime accepts only BF16 weights, so its Qwen3.5 wrapper
    # precomputed (1 + weight) in BF16.
    tle_weight = (1.0 + weight).to(torch.bfloat16).contiguous()

    def reference_rms():
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        return (normalized * (1.0 + weight.float())).to(x.dtype)

    def triton_rms():
        rms_module._rms_norm_qwen35_kernel[(1,)](
            x,
            weight,
            out,
            D=args.d,
            eps=eps,
            BLOCK_SIZE=args.rms_block,
            num_warps=1,
            num_stages=1,
        )
        return out

    def tle_rms():
        tle_rms_function(
            x.data_ptr(),
            tle_weight.data_ptr(),
            tle_out.data_ptr(),
            args.d,
            eps,
        )
        return tle_out

    expected = reference_rms()
    torch.testing.assert_close(triton_rms(), expected)

    gated_x = torch.randn(
        args.gated_m, args.gated_d, dtype=torch.bfloat16
    )
    gate = torch.randn_like(gated_x)
    gated_weight = torch.randn(args.gated_d, dtype=torch.float32) * 0.1
    gated_out = torch.empty_like(gated_x)
    tle_gated_out = torch.empty_like(gated_x)
    tle_gated_weight = gated_weight.to(torch.bfloat16).contiguous()
    tle_gated_function = runtime.standalone_rms_norm_gated_bf16
    tle_gated_function.argtypes = [ctypes.c_void_p] * 4 + [
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_float,
    ]
    tle_gated_function.restype = None

    def reference_gated():
        xf = gated_x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        normalized = normalized.to(gated_x.dtype)
        return (
            gated_weight * normalized * torch.nn.functional.silu(gate.float())
        ).to(gated_x.dtype)

    def triton_gated():
        gated_module._rms_norm_gated_kernel[(args.gated_m,)](
            gated_x,
            gate,
            gated_weight,
            gated_out,
            D=args.gated_d,
            eps=eps,
            BLOCK_SIZE=args.gated_block,
            num_warps=1,
            num_stages=1,
        )
        return gated_out

    def tle_gated():
        tle_gated_function(
            gated_x.data_ptr(),
            gate.data_ptr(),
            tle_gated_weight.data_ptr(),
            tle_gated_out.data_ptr(),
            args.gated_m,
            args.gated_d,
            eps,
        )
        return tle_gated_out

    expected_gated = reference_gated()
    torch.testing.assert_close(triton_gated(), expected_gated)
    tle_rms_error = (tle_rms().float() - expected.float()).abs().max().item()
    tle_gated_error = (
        tle_gated().float() - expected_gated.float()
    ).abs().max().item()

    rms_ref_us = median_us(reference_rms, args.iterations, args.batches)
    rms_triton_us = median_us(triton_rms, args.iterations, args.batches)
    rms_tle_us = median_us(tle_rms, args.iterations, args.batches)
    gated_ref_us = median_us(reference_gated, args.iterations, args.batches)
    gated_triton_us = median_us(triton_gated, args.iterations, args.batches)
    gated_tle_us = median_us(tle_gated, args.iterations, args.batches)
    rms_lines, rms_stack, rms_calls = audit(rms_module._rms_norm_qwen35_kernel)
    gated_lines, gated_stack, gated_calls = audit(
        gated_module._rms_norm_gated_kernel
    )
    print(
        "PASS Qwen3.5 ordinary RMS codegen\n"
        f"rms_d={args.d} reference_us={rms_ref_us:.3f} "
        f"triton_us={rms_triton_us:.3f} "
        f"triton_over_reference={rms_triton_us / rms_ref_us:.3f}x\n"
        f"rms_tle_us={rms_tle_us:.3f} "
        f"triton_over_tle={rms_triton_us / rms_tle_us:.3f}x "
        f"tle_max_abs_error={tle_rms_error:.6f}\n"
        f"gated_shape={args.gated_m}x{args.gated_d} "
        f"reference_us={gated_ref_us:.3f} triton_us={gated_triton_us:.3f} "
        f"triton_over_reference={gated_triton_us / gated_ref_us:.3f}x\n"
        f"gated_tle_us={gated_tle_us:.3f} "
        f"triton_over_tle={gated_triton_us / gated_tle_us:.3f}x "
        f"tle_max_abs_error={tle_gated_error:.6f}\n"
        f"rms_asm_lines={rms_lines} rms_stack={rms_stack} rms_calls={rms_calls}\n"
        f"gated_asm_lines={gated_lines} gated_stack={gated_stack} "
        f"gated_calls={gated_calls}"
    )


if __name__ == "__main__":
    main()
