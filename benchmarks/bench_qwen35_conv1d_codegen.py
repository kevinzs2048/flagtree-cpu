#!/usr/bin/env python3
"""Microbenchmark ordinary-Triton Qwen3.5 causal conv decode update."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import statistics
import sys
import time

import torch
import torch.nn.functional as F


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument(
        "--tle-runtime",
        default="/home/cix/triton-cpu-int8/triton/_C/libTritonCPURuntime.so",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(3504)

    module = importlib.import_module(
        "flag_gems.runtime.backend._arm.fused.patch_qwen3_5_conv1d"
    )
    channels = args.channels
    hidden = torch.randn(1, channels, 1, dtype=torch.bfloat16)
    initial_state = torch.randn(1, channels, 4, dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(channels, dtype=torch.bfloat16) * 0.01
    reference_bias = None if args.no_bias else bias

    reference_state = initial_state.clone()
    triton_state = initial_state.clone()
    tle_state = initial_state.clone()
    out = torch.empty(1, channels, dtype=torch.bfloat16)
    tle_out = torch.empty_like(out)

    runtime = ctypes.CDLL(args.tle_runtime)
    tle_function = runtime.standalone_causal_conv1d_update_bf16
    tle_function.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_long] * 5
    tle_function.restype = None

    def reference():
        combined = torch.cat((reference_state, hidden), dim=-1).to(weight.dtype)
        reference_state.copy_(combined[:, :, -4:])
        value = F.conv1d(
            combined,
            weight.unsqueeze(1),
            reference_bias,
            groups=channels,
        )
        return F.silu(value[:, :, -1:]).to(hidden.dtype)

    def ordinary():
        module._causal_conv1d_update_kernel[(1,)](
            hidden.squeeze(-1),
            triton_state,
            weight,
            bias,
            out,
            B=1,
            C=channels,
            kernel_size=4,
            has_bias=not args.no_bias,
            BLOCK_SIZE=args.block_size,
            num_warps=1,
            num_stages=1,
        )
        return out.unsqueeze(-1)

    def tle_runtime():
        tle_function(
            hidden.data_ptr(),
            tle_state.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            tle_out.data_ptr(),
            1,
            channels,
            4,
            1,
            0 if args.no_bias else 1,
        )
        return tle_out.unsqueeze(-1)

    expected = reference()
    actual = ordinary()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(triton_state, reference_state, rtol=0, atol=0)
    max_error = (actual.float() - expected.float()).abs().max().item()
    tle_actual = tle_runtime()
    torch.testing.assert_close(tle_actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(tle_state, reference_state, rtol=0, atol=0)
    tle_max_error = (tle_actual.float() - expected.float()).abs().max().item()

    # Each timed invocation begins with an already-valid rolling state; both
    # paths update it in place in the same way.
    reference_us = median_us(reference, args.iterations, args.batches)
    ordinary_us = median_us(ordinary, args.iterations, args.batches)
    tle_us = median_us(tle_runtime, args.iterations, args.batches)
    compiled = last_compiled(module._causal_conv1d_update_kernel)
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
        "PASS Qwen3.5 causal conv ordinary codegen\n"
        f"shape=1x{channels}x1 reference_us={reference_us:.3f} "
        f"triton_us={ordinary_us:.3f} "
        f"triton_over_reference={ordinary_us / reference_us:.3f}x\n"
        f"tle_runtime_us={tle_us:.3f} triton_over_tle={ordinary_us / tle_us:.3f}x\n"
        f"max_abs_error={max_error:.6f} tle_max_abs_error={tle_max_error:.6f} "
        f"asm_lines={len(assembly.splitlines())} "
        f"stack_refs={stack} calls={calls}"
    )


if __name__ == "__main__":
    main()
