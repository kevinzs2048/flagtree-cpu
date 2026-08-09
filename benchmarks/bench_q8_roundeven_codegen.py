#!/usr/bin/env python3
"""Compile/audit legacy RNE versus the standard CPU roundeven path.

The companion C++ benchmark calls both cached kernel bodies directly, so the
reported comparison excludes the Python and Triton launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
VENV_SITE = Path(
    os.getenv(
        "TRITON_CPU_VENV_SITE",
        "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
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
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.language.extra import libdevice  # noqa: E402

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    _quantize_bf16_w8_rne_kernel,
)


@triton.jit
def _roundeven_probe_kernel(src_ptr, dst_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    values = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + offsets, libdevice.rint(values))


@triton.jit
def _quantize_bf16_w8_rne_legacy_kernel(
    x_ptr,
    q_ptr,
    x_scale_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Retired arithmetic expansion retained only as an A/B baseline."""
    absmax = tl.zeros((1,), dtype=tl.float32)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        values = tl.load(
            x_ptr + cols, mask=cols < K, other=0.0
        ).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(values), axis=0))
    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inverse = 127.0 / absmax
    tl.store(x_scale_ptr + tl.arange(0, 1), scale)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        values = tl.load(
            x_ptr + cols, mask=cols < K, other=0.0
        ).to(tl.float32)
        scaled = values * inverse
        truncated = scaled.to(tl.int32)
        base = truncated.to(tl.float32)
        fraction = tl.abs(scaled - base)
        adjust = (fraction > 0.5) | (
            (fraction == 0.5) & ((truncated & 1) != 0)
        )
        direction = tl.where(scaled >= 0.0, 1, -1)
        quantized = tl.where(
            adjust, truncated + direction, truncated
        ).to(tl.int8)
        tl.store(q_ptr + cols, quantized, mask=cols < K)


@triton.jit
def _quantize_bf16_w8_rne_float_absmax_kernel(
    x_ptr,
    q_ptr,
    x_scale_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Retired floating-point absmax with the current standard RNE."""
    absmax = tl.zeros((1,), dtype=tl.float32)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        values = tl.load(
            x_ptr + cols, mask=cols < K, other=0.0
        ).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(values), axis=0))
    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inverse = 127.0 / absmax
    tl.store(x_scale_ptr + tl.arange(0, 1), scale)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        values = tl.load(
            x_ptr + cols, mask=cols < K, other=0.0
        ).to(tl.float32)
        quantized = libdevice.rint(values * inverse).to(tl.int8)
        tl.store(q_ptr + cols, quantized, mask=cols < K)


def audit(compiled) -> dict[str, int]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    return {
        "asm_lines": len(assembly.splitlines()),
        "frintn": sum(
            line.lstrip().startswith("frintn")
            for line in assembly.splitlines()
        ),
        "fcvtns": sum(
            line.lstrip().startswith("fcvtns")
            for line in assembly.splitlines()
        ),
        "umax": sum(
            line.lstrip().startswith("umax")
            for line in assembly.splitlines()
        ),
        "fmaxnm": sum(
            line.lstrip().startswith("fmaxnm")
            for line in assembly.splitlines()
        ),
        "folded_spills": assembly.count("folded spill"),
        "folded_reloads": assembly.count("folded reload"),
        "external_calls": sum(
            " call " in line and "llvm." not in line and " asm " not in line
            for line in llir.splitlines()
        ),
    }


def compiled_so(cache: Path, symbol: str) -> Path:
    matches = list(cache.glob(f"**/{symbol}.so"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {symbol}.so under {cache}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batches", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--cpp-benchmark",
        type=Path,
        default=ROOT / "artifacts/bench_q8_roundeven_aot",
    )
    args = parser.parse_args()
    if args.k <= 0 or args.k % 16 or args.block_k != 16:
        raise ValueError("current audit requires positive K divisible by 16")
    cache = Path(os.environ.get("TRITON_CACHE_DIR", ""))
    if not cache:
        raise ValueError("set TRITON_CACHE_DIR to a fresh explicit directory")

    torch.manual_seed(8461)
    probe_values = [
        -7.5,
        -6.5,
        -3.5,
        -2.5,
        -1.5,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        1.5,
        2.5,
        3.5,
        6.5,
        7.5,
        8.25,
    ]
    checked_dtypes = []
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        probe = torch.tensor(probe_values, dtype=dtype)
        probe_out = torch.empty_like(probe)
        _roundeven_probe_kernel[(1,)](
            probe, probe_out, N=16, num_warps=1, num_stages=1
        )
        assert torch.equal(probe_out, probe.round())
        checked_dtypes.append(str(dtype))

    x = torch.randn(args.k, dtype=torch.bfloat16)
    legacy_q = torch.empty(args.k, dtype=torch.int8)
    roundeven_q = torch.empty_like(legacy_q)
    float_abs_q = torch.empty_like(legacy_q)
    legacy_scale = torch.empty(1, dtype=torch.float32)
    roundeven_scale = torch.empty_like(legacy_scale)
    float_abs_scale = torch.empty_like(legacy_scale)

    common = {
        "K": args.k,
        "BLOCK_K": args.block_k,
        "num_warps": 1,
        "num_stages": 1,
        "grid": (1,),
    }
    legacy = _quantize_bf16_w8_rne_legacy_kernel.warmup(
        x, legacy_q, legacy_scale, **common
    )
    roundeven = _quantize_bf16_w8_rne_kernel.warmup(
        x, roundeven_q, roundeven_scale, **common
    )
    float_abs = _quantize_bf16_w8_rne_float_absmax_kernel.warmup(
        x, float_abs_q, float_abs_scale, **common
    )
    _quantize_bf16_w8_rne_legacy_kernel[(1,)](
        x,
        legacy_q,
        legacy_scale,
        K=args.k,
        BLOCK_K=args.block_k,
        num_warps=1,
        num_stages=1,
    )
    _quantize_bf16_w8_rne_kernel[(1,)](
        x,
        roundeven_q,
        roundeven_scale,
        K=args.k,
        BLOCK_K=args.block_k,
        num_warps=1,
        num_stages=1,
    )
    _quantize_bf16_w8_rne_float_absmax_kernel[(1,)](
        x,
        float_abs_q,
        float_abs_scale,
        K=args.k,
        BLOCK_K=args.block_k,
        num_warps=1,
        num_stages=1,
    )
    reference_absmax = x.float().abs().max().clamp(min=1.0e-8)
    reference_q = (
        (x.float() * (127.0 / reference_absmax)).round().to(torch.int8)
    )
    assert torch.equal(legacy_q, roundeven_q)
    assert torch.equal(roundeven_q, reference_q)
    assert torch.equal(float_abs_q, reference_q)
    assert torch.equal(legacy_scale, roundeven_scale)
    assert torch.equal(float_abs_scale, roundeven_scale)

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
    _quantize_bf16_w8_rne_float_absmax_kernel[(1,)](
        x,
        float_abs_q,
        float_abs_scale,
        K=args.k,
        BLOCK_K=args.block_k,
        num_warps=1,
        num_stages=1,
    )
    _quantize_bf16_w8_rne_kernel[(1,)](
        x,
        roundeven_q,
        roundeven_scale,
        K=args.k,
        BLOCK_K=args.block_k,
        num_warps=1,
        num_stages=1,
    )
    assert torch.equal(float_abs_q, roundeven_q)
    assert torch.equal(float_abs_scale, roundeven_scale)

    legacy_audit = audit(legacy)
    roundeven_audit = audit(roundeven)
    float_abs_audit = audit(float_abs)
    assert legacy_audit["frintn"] == 0
    roundeven_rne = roundeven_audit["frintn"] + roundeven_audit["fcvtns"]
    float_abs_rne = float_abs_audit["frintn"] + float_abs_audit["fcvtns"]
    assert roundeven_rne >= 4
    assert roundeven_rne % 4 == 0
    assert roundeven_audit["external_calls"] == 0
    assert roundeven_audit["folded_spills"] == 0
    assert roundeven_audit["folded_reloads"] == 0
    if args.k % 32 == 0:
        assert roundeven_audit["umax"] >= 3
        assert roundeven_audit["umax"] % 3 == 0
        assert (
            roundeven_audit["fmaxnm"]
            == roundeven_audit["umax"] // 3 + 1
        )
    assert float_abs_rne == roundeven_rne
    assert float_abs_audit["external_calls"] == 0
    assert float_abs_audit["folded_spills"] == 0
    assert float_abs_audit["folded_reloads"] == 0

    command = [
        str(args.cpp_benchmark),
        str(compiled_so(cache, "_quantize_bf16_w8_rne_legacy_kernel")),
        str(compiled_so(cache, "_quantize_bf16_w8_rne_kernel")),
        str(args.k),
        str(args.iters),
        str(args.batches),
        str(args.warmup),
    ]
    measured = subprocess.run(
        command, check=True, text=True, capture_output=True
    ).stdout.strip()
    absmax_command = [
        str(args.cpp_benchmark),
        str(compiled_so(cache, "_quantize_bf16_w8_rne_float_absmax_kernel")),
        str(compiled_so(cache, "_quantize_bf16_w8_rne_kernel")),
        str(args.k),
        str(args.iters),
        str(args.batches),
        str(args.warmup),
        "_quantize_bf16_w8_rne_float_absmax_kernel",
        "_quantize_bf16_w8_rne_kernel",
    ]
    absmax_measured = subprocess.run(
        absmax_command, check=True, text=True, capture_output=True
    ).stdout.strip()
    print(
        json.dumps(
            {
                "status": "PASS",
                "triton_module": triton.__file__,
                "k": args.k,
                "bit_exact": True,
                "finite_bf16_edges_bit_exact": True,
                "roundeven_dtypes": checked_dtypes,
                "legacy_codegen": legacy_audit,
                "roundeven_codegen": roundeven_audit,
                "float_absmax_codegen": float_abs_audit,
            },
            indent=2,
        )
    )
    print(measured)
    print(absmax_measured)


if __name__ == "__main__":
    main()
