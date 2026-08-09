#!/usr/bin/env python3
"""Profile compiler-generated W8 SwiGLU against the legacy fused C runtime.

The Triton kernel is intentionally only the gate/up/SwiGLU half of a Qwen MLP.
The down projection is benchmarked separately by the decode GEMV benchmark, so
the comparison below has the same inputs, outputs and amount of work.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time

import torch
import torch.nn.functional as F

from flag_gems.runtime.backend._arm.fused.patch_qwen3_mlp import (
    _fused_mlp_kernel,
)
from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    TLEInt8Linear,
    pack_weights_sdot,
    pack_weights_sdot_blocked,
    retile_weights_sdot_blocked,
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
    parser.add_argument("--block-n", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument(
        "--runtime-so",
        default="/home/kevin/triton-opt-cpu/python/triton/_C/"
        "libTritonCPURuntime.so",
    )
    args = parser.parse_args()
    if args.k % 4 or args.n % args.block_n or args.block_n % 4:
        raise ValueError("K%4, N%BLOCK_N and BLOCK_N%4 must be zero")

    torch.manual_seed(31)
    x = (torch.randn(args.k) * 0.2).to(torch.bfloat16)
    gate_weight = torch.randint(-127, 128, (args.n, args.k), dtype=torch.int8)
    up_weight = torch.randint(-127, 128, (args.n, args.k), dtype=torch.int8)
    gate_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    up_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    gate = TLEInt8Linear(gate_weight, gate_scale)
    up = TLEInt8Linear(up_weight, up_scale)

    # Production releases the redundant row-major INT8 copy after creating its
    # packed layouts.  Reconstruct it lazily here because this benchmark also
    # needs the legacy K-major runtime pack for the A/B comparison.
    gate_kmajor = pack_weights_sdot(gate._get_w_int8_kn())
    up_kmajor = pack_weights_sdot(up._get_w_int8_kn())
    gate_codegen = pack_weights_sdot_blocked(gate_kmajor, args.block_n)
    up_codegen = pack_weights_sdot_blocked(up_kmajor, args.block_n)
    gate_codegen = retile_weights_sdot_blocked(
        gate_codegen, args.block_n, 32
    )
    up_codegen = retile_weights_sdot_blocked(
        up_codegen, args.block_n, 32
    )
    codegen_out = torch.empty(args.n, dtype=torch.bfloat16)
    runtime_out = torch.empty_like(codegen_out)

    def run_codegen() -> None:
        _fused_mlp_kernel[(args.n // args.block_n,)](
            x,
            gate_codegen,
            up_codegen,
            gate._w_scale,
            up._w_scale,
            codegen_out,
            K=args.k,
            N=args.n,
            BLOCK_N=args.block_n,
        )

    runtime = ctypes.CDLL(args.runtime_so)
    runtime_mlp = runtime.fused_mlp_bf16
    runtime_mlp.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 2
    runtime_mlp.restype = None

    def run_runtime() -> None:
        runtime_mlp(
            x.data_ptr(),
            gate_kmajor.data_ptr(),
            up_kmajor.data_ptr(),
            gate._w_scale.data_ptr(),
            up._w_scale.data_ptr(),
            runtime_out.data_ptr(),
            args.k,
            args.n,
        )

    run_codegen()
    gate_ref = gate(x.reshape(1, -1))
    up_ref = up(x.reshape(1, -1))
    reference = (F.silu(gate_ref) * up_ref).reshape(-1)
    if not torch.equal(codegen_out, reference):
        delta = (codegen_out.float() - reference.float()).abs()
        raise AssertionError(
            "codegen mismatch: "
            f"count={torch.count_nonzero(delta).item()} max={delta.max().item()}"
        )
    run_runtime()

    codegen_us = median_us(
        run_codegen, args.warmup, args.iters, args.batches
    )
    runtime_us = median_us(
        run_runtime, args.warmup, args.iters, args.batches
    )
    runtime_delta = (runtime_out.float() - reference.float()).abs()

    cache = next(iter(_fused_mlp_kernel.device_caches.values()))[0]
    compiled = list(cache.values())[-1]
    llir = compiled.asm["llir"]
    assembly = compiled.asm["asm"].lower()
    print(
        f"PASS K={args.k} N={args.n} BLOCK_N={args.block_n}\n"
        f"triton_kernel_us={codegen_us:.3f}\n"
        f"legacy_whole_c_runtime_us={runtime_us:.3f}\n"
        f"triton_over_c={codegen_us / runtime_us:.3f}x\n"
        f"codegen_bit_exact={torch.equal(codegen_out, reference)}\n"
        f"legacy_c_equal={torch.equal(runtime_out, reference)}\n"
        f"legacy_c_max_abs={runtime_delta.max().item():.8f}\n"
        f"llir_neon_sdot={llir.count('llvm.aarch64.neon.sdot')}\n"
        f"asm_sdot={assembly.count('sdot')}\n"
        f"llir_exp={llir.count('llvm.exp') + llir.count('math.exp')}\n"
        f"llir_external_fused_runtime={llir.count('fused_mlp_bf16')}\n"
        f"asm_vector_exp_calls={assembly.count('sleef_expf4')}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_folded_spills={assembly.count('folded spill')}\n"
        f"asm_folded_reloads={assembly.count('folded reload')}"
    )


if __name__ == "__main__":
    main()
