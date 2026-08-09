#!/usr/bin/env python3
"""Profile compiler-generated packed W8 decode against the legacy C runtime."""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time

import torch

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    TLEInt8Linear,
    _tle_fused_bf16_gemv_kernel,
    _tle_whole_bf16_gemv_kernel,
    pack_weights_sdot,
    pack_weights_sdot_blocked,
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
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--whole",
        action="store_true",
        help="quantize once and roll all N/TILE_N microtiles in one program",
    )
    parser.add_argument(
        "--runtime-so",
        default="/home/kevin/triton-opt-cpu/python/triton/_C/"
        "libTritonCPURuntime.so",
    )
    args = parser.parse_args()
    if args.k % 4 or args.n % args.block_n or args.block_n % 4:
        raise ValueError("K%4, N%BLOCK_N and BLOCK_N%4 must be zero")

    torch.manual_seed(29)
    x = torch.randn(args.k, dtype=torch.bfloat16)
    weight_nk = torch.randint(
        -127, 128, (args.n, args.k), dtype=torch.int8
    )
    weight_kn = weight_nk.T.contiguous()
    weight_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    linear = TLEInt8Linear(weight_nk, weight_scale)
    packed_kmajor = pack_weights_sdot(weight_kn)
    packed_codegen = pack_weights_sdot_blocked(packed_kmajor, args.block_n)
    codegen_out = torch.empty(args.n, dtype=torch.bfloat16)
    runtime_out = torch.empty_like(codegen_out)

    def run_codegen() -> None:
        if args.whole:
            _tle_whole_bf16_gemv_kernel[(1,)](
                x,
                packed_codegen,
                linear._w_scale,
                codegen_out,
                K=args.k,
                N=args.n,
                TILE_N=args.block_n,
            )
        else:
            _tle_fused_bf16_gemv_kernel[(args.n // args.block_n,)](
                x,
                packed_codegen,
                linear._w_scale,
                codegen_out,
                K=args.k,
                N=args.n,
                BLOCK_N=args.block_n,
            )

    runtime = ctypes.CDLL(args.runtime_so)
    runtime_gemv = runtime.sdot_gemv_m1_fused_bf16
    runtime_gemv.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3
    runtime_gemv.restype = None

    def run_runtime() -> None:
        runtime_gemv(
            x.data_ptr(),
            packed_kmajor.data_ptr(),
            linear._w_scale.data_ptr(),
            runtime_out.data_ptr(),
            args.k,
            args.n,
            args.n // 4,
        )

    def reference_for(values: torch.Tensor) -> torch.Tensor:
        values_f32 = values.float()
        absmax = values_f32.abs().amax().clamp(min=1.0e-8)
        x_scale = absmax / 127.0
        q = (
            (values_f32 * (127.0 / absmax))
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )
        return (
            torch._int_mm(q.reshape(1, -1), weight_kn).float()
            * x_scale
            * linear._w_scale
        ).to(torch.bfloat16).reshape(-1)

    run_codegen()
    run_runtime()
    reference = reference_for(x)
    selected_kernel = (
        _tle_whole_bf16_gemv_kernel
        if args.whole
        else _tle_fused_bf16_gemv_kernel
    )
    cache = next(iter(selected_kernel.device_caches.values()))[0]
    compiled = list(cache.values())[-1]
    llir = compiled.asm["llir"]
    assembly = compiled.asm["asm"].lower()
    is_inline = "sdot_gemv_m1_fused_bf16" not in llir
    if is_inline and not torch.equal(codegen_out, reference):
        delta = (codegen_out.float() - reference.float()).abs()
        raise AssertionError(
            "codegen mismatch: "
            f"count={torch.count_nonzero(delta).item()} max={delta.max().item()}"
        )

    # Finite BF16 magnitude bits are monotonic after clearing the sign bit,
    # which is the precondition used by the lane-max compiler lowering.
    normal_x = x.clone()
    edge_bits = torch.tensor(
        [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F80,
            0xBF80,
            0x3F81,
            0xBF81,
            0x7F7F,
            0xFF7F,
        ],
        dtype=torch.uint16,
    )
    repeats = (args.k + edge_bits.numel() - 1) // edge_bits.numel()
    x.copy_(edge_bits.repeat(repeats)[: args.k].view(torch.bfloat16))
    run_codegen()
    edge_reference = reference_for(x)
    edge_bit_exact = torch.equal(codegen_out, edge_reference)
    if is_inline and not edge_bit_exact:
        raise AssertionError("finite BF16 edge-pattern codegen mismatch")
    x.copy_(normal_x)
    run_codegen()

    codegen_us = median_us(
        run_codegen, args.warmup, args.iters, args.batches
    )
    runtime_us = median_us(
        run_runtime, args.warmup, args.iters, args.batches
    )
    runtime_delta = (runtime_out.float() - reference.float()).abs()
    print(
        f"PASS K={args.k} N={args.n} "
        f"{'TILE_N' if args.whole else 'BLOCK_N'}={args.block_n}\n"
        f"codegen_mode={'whole_microtile' if args.whole else 'grid_blocks'}\n"
        f"triton_kernel_us={codegen_us:.3f}\n"
        f"legacy_whole_c_runtime_us={runtime_us:.3f}\n"
        f"triton_over_c={codegen_us / runtime_us:.3f}x\n"
        f"codegen_bit_exact={torch.equal(codegen_out, reference)}\n"
        f"finite_bf16_edges_bit_exact={edge_bit_exact}\n"
        f"legacy_c_max_abs={runtime_delta.max().item():.8f}\n"
        f"llir_neon_sdot={llir.count('llvm.aarch64.neon.sdot')}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"llir_external_fused_runtime="
        f"{llir.count('sdot_gemv_m1_fused_bf16')}"
    )


if __name__ == "__main__":
    main()
