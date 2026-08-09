#!/usr/bin/env python3
"""Compare independent and fused QKV ordinary-Triton decode codegen."""

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

import torch  # noqa: E402

from flag_gems.runtime.backend._arm.fused.patch_qwen3_qkv import (  # noqa: E402
    patch_qwen3_qkv,
)
from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    TLEInt8Linear,
)


class AttentionProjections(torch.nn.Module):
    def __init__(self, weights, scales) -> None:
        super().__init__()
        self.q_proj = TLEInt8Linear(weights[0], scales[0])
        self.k_proj = TLEInt8Linear(weights[1], scales[1])
        self.v_proj = TLEInt8Linear(weights[2], scales[2])

    def forward(self, x: torch.Tensor):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)


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
    parser.add_argument("--bundle")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--whole-mode", choices=("auto", "0", "1"), default="auto"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--batches", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(816)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ["FLAGGEMS_ARM_QKV_WHOLE_CODEGEN"] = args.whole_mode
    k = 1024
    output_sizes = (2048, 1024, 1024)
    x = torch.randn(k, dtype=torch.bfloat16)
    weights = tuple(
        torch.randint(-127, 128, (n, k), dtype=torch.int8)
        for n in output_sizes
    )
    scales = tuple(
        torch.rand(n, dtype=torch.float32) / 127.0 for n in output_sizes
    )

    os.environ.pop("FLAGGEMS_ARM_W8_AOT_BUNDLE", None)
    independent = AttentionProjections(weights, scales)
    jit = AttentionProjections(weights, scales)
    assert patch_qwen3_qkv(jit) == 1
    assert jit._triton_qkv_coordinator.ordinary_aot is None

    aot = None
    if args.bundle:
        os.environ["FLAGGEMS_ARM_W8_AOT_BUNDLE"] = args.bundle
        aot = AttentionProjections(weights, scales)
        assert patch_qwen3_qkv(aot) == 1
        assert aot._triton_qkv_coordinator.ordinary_aot is not None

    with torch.inference_mode():
        independent_output = independent(x)
        jit_output = jit(x)
        assert all(
            torch.equal(independent_part, fused_part)
            for independent_part, fused_part in zip(
                independent_output, jit_output
            )
        )
        independent_us = median_us(
            lambda: independent(x),
            args.warmup,
            args.iters,
            args.batches,
        )
        jit_us = median_us(
            lambda: jit(x), args.warmup, args.iters, args.batches
        )
        aot_us = None
        if aot is not None:
            aot_output = aot(x)
            assert all(
                torch.equal(aot_part, fused_part)
                for aot_part, fused_part in zip(aot_output, jit_output)
            )
            aot_us = median_us(
                lambda: aot(x), args.warmup, args.iters, args.batches
            )

    lines = [
        "PASS FlagGems fused BF16-W8 QKV",
        f"threads={args.threads}",
        f"whole_mode={args.whole_mode}",
        f"independent_qkv_us={independent_us:.3f}",
        f"ordinary_jit_qkv_us={jit_us:.3f}",
        f"fused_over_independent={jit_us / independent_us:.3f}x",
    ]
    if aot_us is not None:
        lines.extend(
            [
                f"ordinary_aot_qkv_us={aot_us:.3f}",
                f"aot_over_jit={aot_us / jit_us:.3f}x",
            ]
        )
    lines.append("bit_exact=True")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
