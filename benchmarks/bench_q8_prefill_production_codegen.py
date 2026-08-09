#!/usr/bin/env python3
"""Production W8 short-prefill A/B: row-major N64 versus KAI N4/K8."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
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


def require_expected_triton() -> None:
    expected = (TRITON_PYTHON / "triton").resolve()
    actual = Path(triton.__file__).resolve()
    if expected not in actual.parents:
        raise RuntimeError(
            "wrong Triton source loaded: "
            f"expected under {expected}, got {actual}. "
            "Put TRITON_CPU_PYTHON at the front of PYTHONPATH before "
            "starting Python; setting it only inside the process is too late "
            "when a site .pth file preloads Triton."
        )


require_expected_triton()

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    _fused_prefill_w8a8,
    _int8mm_dequant_bf16_kernel,
    _pack_lhs_w8_i8mm_kai_kernel,
    _w8_prefill_i8mm_kai_m12_kernel,
    _w8_prefill_i8mm_kai_kernel,
    _w8_prefill_i8mm_kai_short_tail_kernel,
    pack_weights_i8mm_kai,
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


def paired_median_us(
    left, right, warmup: int, iterations: int, batches: int
) -> tuple[float, float]:
    """Alternate A/B order so thermal drift does not favor one schedule."""
    for _ in range(warmup):
        left()
        right()
    samples = [[], []]

    def measure(function) -> float:
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        return (time.perf_counter_ns() - begin) / iterations / 1000.0

    for batch in range(batches):
        order = (0, 1) if batch % 2 == 0 else (1, 0)
        functions = (left, right)
        for index in order:
            samples[index].append(measure(functions[index]))
    return statistics.median(samples[0]), statistics.median(samples[1])


def audit(compiled) -> dict[str, int | bool]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    return {
        "asm_lines": len(assembly.splitlines()),
        "smmla": assembly.count("smmla"),
        "folded_spills": assembly.count("folded spill"),
        "folded_reloads": assembly.count("folded reload"),
        "external_calls": sum(
            " call " in line and "llvm." not in line and " asm " not in line
            for line in llir.splitlines()
        ),
        "residual_triton_dot": "triton_cpu.dot" in llir,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batches", type=int, default=9)
    args = parser.parse_args()
    if not 2 <= args.m <= 1024 or args.n % 64 or args.k % 32:
        raise ValueError("requires 2<=M<=1024, N%64=0 and K%32=0")

    torch.set_num_threads(1)
    torch.manual_seed(8927)
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    weight_nk = torch.randint(
        -127, 128, (args.n, args.k), dtype=torch.int8
    )
    weight_kn = weight_nk.T.contiguous()
    weight_scale = torch.rand(args.n, dtype=torch.float32) / 127.0
    weight_kai = pack_weights_i8mm_kai(weight_nk, weight_scale)

    def run_legacy():
        return _fused_prefill_w8a8(x, weight_kn, weight_scale)

    def run_kai():
        return _fused_prefill_w8a8(
            x, weight_kn, weight_scale, weight_kai
        )

    def run_kai_padded_m16():
        """Retired production schedule: round every short tail to M16."""
        shape = x.shape
        runtime_k = shape[-1]
        runtime_m = x.numel() // runtime_k
        stride = 4 * runtime_k + 16
        runtime_n = (weight_kai.numel() // stride) * 4
        xf = x.reshape(runtime_m, runtime_k).contiguous()
        padded_m = ((runtime_m + 15) // 16) * 16
        packed_lhs = torch.empty(
            (padded_m // 4) * stride, dtype=torch.int8
        )
        _pack_lhs_w8_i8mm_kai_kernel[(padded_m,)](
            xf,
            packed_lhs,
            runtime_m,
            xf.stride(0),
            K=runtime_k,
            num_warps=1,
            num_stages=1,
        )
        output = torch.empty(
            (padded_m, runtime_n), dtype=torch.bfloat16
        )
        _w8_prefill_i8mm_kai_kernel[(padded_m // 16, runtime_n // 4)](
            packed_lhs,
            weight_kai,
            output,
            N=runtime_n,
            K=runtime_k,
            BLOCK_M=16,
            num_warps=1,
            num_stages=1,
        )
        return output[:runtime_m].reshape(*shape[:-1], runtime_n)

    legacy = run_legacy()
    kai = run_kai()
    kai_padded_m16 = run_kai_padded_m16()
    assert legacy is not None and kai is not None
    assert torch.equal(legacy, kai)
    assert torch.equal(kai_padded_m16, kai)

    short_tail = args.m <= 12
    m_kai = (
        (4 if args.m <= 4 else 8 if args.m <= 8 else 12)
        if short_tail
        else ((args.m + 15) // 16) * 16
    )
    panel_stride = 4 * args.k + 16
    lhs = torch.empty((m_kai // 4) * panel_stride, dtype=torch.int8)
    out_kai = torch.empty((m_kai, args.n), dtype=torch.bfloat16)
    pack_compiled = _pack_lhs_w8_i8mm_kai_kernel.warmup(
        x,
        lhs,
        args.m,
        x.stride(0),
        K=args.k,
        FULL_ROWS=args.m == m_kai,
        num_warps=1,
        num_stages=1,
        grid=(m_kai,),
    )
    kai_compiled = []
    if not short_tail:
        kai_compiled.append((
            "main_m16",
            _w8_prefill_i8mm_kai_kernel.warmup(
                lhs,
                weight_kai,
                out_kai,
                N=args.n,
                K=args.k,
                BLOCK_M=16,
                num_warps=1,
                num_stages=1,
                grid=(m_kai // 16, args.n // 4),
            ),
        ))
    elif m_kai <= 8:
        kai_compiled.append((
            f"short_m{m_kai}",
            _w8_prefill_i8mm_kai_short_tail_kernel.warmup(
                lhs,
                weight_kai,
                out_kai,
                N=args.n,
                K=args.k,
                BLOCK_M=m_kai,
                num_warps=1,
                num_stages=1,
                grid=(1, args.n // 4),
            ),
        ))
    else:
        kai_compiled.append((
            "short_m12",
            _w8_prefill_i8mm_kai_m12_kernel.warmup(
                lhs,
                weight_kai,
                out_kai,
                N=args.n,
                K=args.k,
                num_warps=1,
                num_stages=1,
                grid=(1, args.n // 4),
            ),
        ))

    m_legacy = ((args.m + 7) // 8) * 8
    q = torch.empty((m_legacy, args.k), dtype=torch.int8)
    x_scale = torch.empty(m_legacy, dtype=torch.float32)
    out_legacy = torch.empty((m_legacy, args.n), dtype=torch.bfloat16)
    legacy_compiled = _int8mm_dequant_bf16_kernel.warmup(
        q,
        weight_kn,
        x_scale,
        weight_scale,
        out_legacy,
        args.m,
        args.n,
        args.k,
        q.stride(0),
        q.stride(1),
        weight_kn.stride(0),
        weight_kn.stride(1),
        out_legacy.stride(0),
        out_legacy.stride(1),
        BLOCK_M=8,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=1,
        num_stages=1,
        grid=(m_legacy // 8, args.n // 64),
    )

    pack_codegen = audit(pack_compiled)
    kai_codegen = {name: audit(compiled) for name, compiled in kai_compiled}
    legacy_codegen = audit(legacy_compiled)
    assert pack_codegen["external_calls"] == 0
    assert pack_codegen["folded_spills"] == 0
    for result in kai_codegen.values():
        assert result["smmla"] > 0
        assert result["folded_spills"] <= 2
        assert result["folded_reloads"] <= 2
        assert result["external_calls"] == 0
        assert not result["residual_triton_dot"]

    legacy_us = median_us(
        run_legacy, args.warmup, args.iters, args.batches
    )
    kai_padded_m16_us, kai_us = paired_median_us(
        run_kai_padded_m16,
        run_kai,
        args.warmup,
        args.iters,
        args.batches,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "triton_module": triton.__file__,
                "shape": [args.m, args.k, args.n],
                "bit_exact": True,
                "legacy_row_major_us": legacy_us,
                "retired_kai_padded_m16_us": kai_padded_m16_us,
                "kai_layout_us": kai_us,
                "kai_over_legacy": kai_us / legacy_us,
                "speedup": legacy_us / kai_us,
                "tail_speedup_over_padded_m16": kai_padded_m16_us / kai_us,
                "lhs_pack_codegen": pack_codegen,
                "kai_matrix_codegen": kai_codegen,
                "legacy_matrix_codegen": legacy_codegen,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
