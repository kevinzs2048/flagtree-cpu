#!/usr/bin/env python3
"""Microbenchmark separate versus fused SwiGLU + W8 quantization."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/home/kevin/triton-opt-cpu")
VENV_SITE = Path("/home/cix/venv-fep-e2e/lib/python3.11/site-packages")
TRITON_PYTHON = Path(
    os.getenv("TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python")
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    _quantize_bf16_w8_rne_kernel,
)
from flag_gems.runtime.backend._arm.ops.silu_and_mul import (  # noqa: E402
    _SWIGLU_TILE,
    _swiglu_ordinary_kernel,
    _swiglu_quantize_w8_rne_kernel,
)


def median_us(fn, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        begin = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - begin) / 1e3)
        assert result is not None
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=3072)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=301)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(809)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    gate = torch.randn(args.elements, dtype=torch.bfloat16)
    up = torch.randn(args.elements, dtype=torch.bfloat16)

    if args.compile_only:
        bf16 = torch.empty_like(gate)
        quantized = torch.empty(args.elements, dtype=torch.int8)
        scale = torch.empty(1, dtype=torch.float32)
        compiled = _swiglu_quantize_w8_rne_kernel.warmup(
            gate,
            up,
            bf16,
            quantized,
            scale,
            args.elements,
            BLOCK_SIZE=_SWIGLU_TILE,
            grid=(1,),
        )
        assembly = compiled.asm["asm"].lower()
        llir = compiled.asm["llir"].lower()
        stack_load_store = sum(
            "[sp" in line and line.lstrip().startswith(("ld", "st"))
            for line in assembly.splitlines()
        )
        external_calls = sum(
            " call " in line and "llvm." not in line and " asm " not in line
            for line in llir.splitlines()
        )
        if external_calls:
            raise RuntimeError(
                "SwiGLU/quant codegen regression: "
                f"external_calls={external_calls}"
            )
        print(
            f"COMPILED fused SwiGLU/W8 quant N={args.elements}\n"
            f"asm_lines={len(assembly.splitlines())}\n"
            f"stack_load_store={stack_load_store}\n"
            f"external_calls={external_calls}"
        )
        return

    def separate():
        bf16 = torch.empty_like(gate)
        quantized = torch.empty(args.elements, dtype=torch.int8)
        scale = torch.empty(1, dtype=torch.float32)
        _swiglu_ordinary_kernel[(1,)](
            gate,
            up,
            bf16,
            args.elements,
            BLOCK_SIZE=_SWIGLU_TILE,
        )
        _quantize_bf16_w8_rne_kernel[(1,)](
            bf16,
            quantized,
            scale,
            K=args.elements,
            BLOCK_K=16,
        )
        return bf16, quantized, scale

    def fused():
        bf16 = torch.empty_like(gate)
        quantized = torch.empty(args.elements, dtype=torch.int8)
        scale = torch.empty(1, dtype=torch.float32)
        _swiglu_quantize_w8_rne_kernel[(1,)](
            gate,
            up,
            bf16,
            quantized,
            scale,
            args.elements,
            BLOCK_SIZE=_SWIGLU_TILE,
        )
        return bf16, quantized, scale

    reference = separate()
    actual = fused()
    exact = all(
        torch.equal(expected, result)
        for expected, result in zip(reference, actual)
    )
    for _ in range(20):
        separate()
        fused()
    separate_us = median_us(separate, args.iterations)
    fused_us = median_us(fused, args.iterations)
    print(
        f"threads={args.threads}\n"
        f"elements={args.elements}\n"
        f"separate_us={separate_us:.3f}\n"
        f"fused_us={fused_us:.3f}\n"
        f"fused_over_separate={fused_us / separate_us:.3f}x\n"
        f"bit_exact={exact}"
    )


if __name__ == "__main__":
    main()
