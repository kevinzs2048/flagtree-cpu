#!/usr/bin/env python3
"""Microbenchmark FlagGems ordinary-JIT and AOT W8 gate/up SwiGLU."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = Path(
    os.getenv(
        "TRITON_CPU_VENV_SITE",
        "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
    )
)
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON",
        ROOT / "ports/triton-cpu-3.7.2/python",
    )
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from flag_gems.runtime.backend._arm.fused.patch_qwen3_mlp import (  # noqa: E402
    FusedMLPWrapper,
)
from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    TLEInt8Linear,
    _quantize_bf16_w8_rne_kernel,
    _w8_decode_sdot_kernel,
)
from flag_gems.runtime.backend._arm.ops.silu_and_mul import (  # noqa: E402
    _SWIGLU_TILE,
    _swiglu_ordinary_kernel,
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


def make_wrapper(gate_weight, up_weight, gate_scale, up_scale):
    wrapper = FusedMLPWrapper(
        TLEInt8Linear(gate_weight, gate_scale),
        TLEInt8Linear(up_weight, up_scale),
        torch.nn.Identity(),
        F.silu,
    )
    assert wrapper._fused
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(867)
    torch.set_num_threads(1)
    x = (torch.randn(args.k) * 0.2).to(torch.bfloat16)
    gate_weight = torch.randint(
        -127, 128, (args.n, args.k), dtype=torch.int8
    )
    up_weight = torch.randint(
        -127, 128, (args.n, args.k), dtype=torch.int8
    )
    gate_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    up_scale = torch.rand(args.n, dtype=torch.float32) / 127.0

    os.environ["FLAGGEMS_ARM_W8_AOT_BUNDLE"] = args.bundle
    aot = make_wrapper(gate_weight, up_weight, gate_scale, up_scale)
    assert aot._ordinary_aot is not None

    os.environ.pop("FLAGGEMS_ARM_W8_AOT_BUNDLE", None)
    jit = make_wrapper(gate_weight, up_weight, gate_scale, up_scale)
    assert jit._ordinary_aot is None

    with torch.inference_mode():
        aot_output = aot.forward(x)
        jit_output = jit.forward(x)
        assert torch.equal(aot_output, jit_output)
        quantized = torch.empty(args.k, dtype=torch.int8)
        activation_scale = torch.empty(1, dtype=torch.float32)
        projection_out = torch.empty(
            2 * jit._N_codegen, dtype=torch.bfloat16
        )
        direct_output = torch.empty(
            jit._N_codegen, dtype=torch.bfloat16
        )

        def direct_pipeline():
            _quantize_bf16_w8_rne_kernel[(1,)](
                x,
                quantized,
                activation_scale,
                K=args.k,
                BLOCK_K=16,
            )
            _w8_decode_sdot_kernel[(1,)](
                quantized,
                activation_scale,
                jit._gate_up_packed,
                jit._gate_up_scale,
                projection_out,
                K=args.k,
                N=2 * jit._N_codegen,
                BLOCK_N=jit._packed_tile_n,
                UNROLL=2,
                WHOLE_PROJECTION=True,
            )
            _swiglu_ordinary_kernel[(1,)](
                projection_out[: jit._N_codegen],
                projection_out[jit._N_codegen :],
                direct_output,
                jit._N_codegen,
                BLOCK_SIZE=_SWIGLU_TILE,
                num_warps=1,
                num_stages=1,
            )

        direct_pipeline()
        assert torch.equal(direct_output[: args.n], aot_output)
        aot_us = median_us(
            lambda: aot.forward(x),
            args.warmup,
            args.iters,
            args.batches,
        )
        jit_us = median_us(
            lambda: jit.forward(x),
            args.warmup,
            args.iters,
            args.batches,
        )
        direct_us = median_us(
            direct_pipeline,
            args.warmup,
            args.iters,
            args.batches,
        )

    print(
        f"PASS FlagGems BF16-W8 SwiGLU K={args.k} N={args.n}\n"
        f"ordinary_aot_mlp_us={aot_us:.3f}\n"
        f"ordinary_jit_mlp_us={jit_us:.3f}\n"
        f"preallocated_jit_pipeline_us={direct_us:.3f}\n"
        f"aot_over_jit={aot_us / jit_us:.3f}x\n"
        f"bit_exact={torch.equal(aot_output, jit_output)}"
    )


if __name__ == "__main__":
    main()
