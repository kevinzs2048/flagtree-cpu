#!/usr/bin/env python3
"""Compile and gate every active CIX Q4/Q8 production specialization."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402
import triton  # noqa: E402


def require_expected_triton() -> None:
    expected = (TRITON_PYTHON / "triton").resolve()
    actual = Path(triton.__file__).resolve()
    if expected not in actual.parents:
        raise RuntimeError(
            "wrong Triton source loaded: "
            f"expected under {expected}, got {actual}"
        )


require_expected_triton()

from flag_gems.runtime.backend._arm.int8 import (  # noqa: E402
    tle_int8_linear as w8,
)
from flag_gems.runtime.backend._arm.q4.kernels import (  # noqa: E402
    _pack_lhs_qsi8d32p_decode_kernel,
    _pack_lhs_qsi8d32p_panel4_scalar_kernel,
    _pack_lhs_qsi8d32p_row_kernel,
    _q4_decode_sdot_kai_kernel,
    _q4_prefill_i8mm_kai_kernel,
)
from flag_gems.runtime.backend._arm.q4.linear import _decode_unroll  # noqa: E402


def instruction_count(assembly: str, opcode: str) -> int:
    return sum(
        bool((fields := line.lstrip().split(None, 1)) and fields[0] == opcode)
        for line in assembly.lower().splitlines()
    )


def inner_loop_stack_metrics(assembly: str) -> tuple[int, int]:
    """Count matched LLVM inner loops and their stack accesses."""
    lines = assembly.lower().splitlines()
    matched_loops = 0
    references = 0
    for start, line in enumerate(lines):
        if "inner loop header" not in line:
            continue
        label = line.split(":", 1)[0].strip()
        if not label:
            continue
        for end in range(start + 1, len(lines)):
            branch = lines[end].strip()
            if branch.startswith("b.") and label in branch:
                matched_loops += 1
                references += sum(
                    "[sp" in body for body in lines[start : end + 1]
                )
                break
    return matched_loops, references


def audit(compiled) -> dict[str, object]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    annotated_inner_loops, loop_stack_refs = inner_loop_stack_metrics(
        assembly
    )
    call_lines = [
        line
        for line in llir.splitlines()
        if " call " in line and "llvm." not in line and " asm " not in line
    ]
    return {
        "asm_lines": len(assembly.splitlines()),
        "sdot": instruction_count(assembly, "sdot"),
        "smmla": instruction_count(assembly, "smmla"),
        "addp": instruction_count(assembly, "addp"),
        "umax": instruction_count(assembly, "umax"),
        "fmaxnm": instruction_count(assembly, "fmaxnm"),
        "fminnm": instruction_count(assembly, "fminnm"),
        "frintn": instruction_count(assembly, "frintn"),
        "fcvtns": instruction_count(assembly, "fcvtns"),
        "folded_spills": assembly.count("folded spill"),
        "folded_reloads": assembly.count("folded reload"),
        "stack_refs": sum("[sp" in line for line in assembly.splitlines()),
        "annotated_inner_loops": annotated_inner_loops,
        "inner_loop_stack_refs": loop_stack_refs,
        "external_calls": sum("sleef_" not in line for line in call_lines),
        "vector_math_calls": sum("sleef_" in line for line in call_lines),
        "residual_triton_dot": "triton_cpu.dot" in llir,
        "neon_sdot_intrinsics": llir.count("llvm.aarch64.neon.sdot"),
        "neon_smmla_intrinsics": llir.count("llvm.aarch64.neon.smmla"),
        "sve_smmla_intrinsics": llir.count("llvm.aarch64.sve.smmla"),
    }


def warmup(kernel, grid, *args, **kwargs):
    return kernel.warmup(
        *args,
        **kwargs,
        num_warps=1,
        num_stages=1,
        grid=grid,
    )


def check_common(result: dict[str, object]) -> None:
    assert result["external_calls"] == 0
    assert not result["residual_triton_dot"]


def check_q8_i8mm(result: dict[str, object]) -> None:
    """Require exactly one scalable or fixed-width SMMLA lowering."""
    sve = result["sve_smmla_intrinsics"] > 0
    neon = result["neon_smmla_intrinsics"] > 0
    assert sve != neon, (sve, neon)
    if os.getenv("TRITON_CPU_FIXED_I8MM", "0").lower() in {
        "1",
        "true",
        "on",
    }:
        assert neon and not sve


def audit_q4(k: int, n: int) -> dict[str, object]:
    groups = k // 32
    x = torch.randn((16, k), dtype=torch.bfloat16)
    lhs = torch.empty(4 * groups * 136, dtype=torch.uint8)
    rhs = torch.empty((n // 4) * groups * 72, dtype=torch.uint8)
    out = torch.empty((16, n), dtype=torch.bfloat16)

    row_pack = audit(
        warmup(
            _pack_lhs_qsi8d32p_row_kernel,
            (16,),
            x,
            lhs.view(torch.float16),
            lhs.view(torch.int8),
            16,
            x.stride(0),
            K=k,
        )
    )
    check_common(row_pack)
    assert row_pack["frintn"] + row_pack["fcvtns"] == 8
    assert row_pack["asm_lines"] < 600
    assert row_pack["folded_spills"] == 0
    assert row_pack["folded_reloads"] == 0

    panel4_pack = audit(
        warmup(
            _pack_lhs_qsi8d32p_panel4_scalar_kernel,
            (4,),
            x,
            lhs.view(torch.float16),
            lhs.view(torch.int8),
            16,
            x.stride(0),
            K=k,
            FULL_PANEL=True,
        )
    )
    check_common(panel4_pack)
    assert panel4_pack["frintn"] + panel4_pack["fcvtns"] == 32
    assert panel4_pack["umax"] == 12
    assert panel4_pack["fmaxnm"] == 0
    assert panel4_pack["fminnm"] == 0
    assert panel4_pack["asm_lines"] < 900
    assert panel4_pack["folded_spills"] == 0
    assert panel4_pack["folded_reloads"] == 0

    decode_lhs = torch.empty(3 * groups * 34, dtype=torch.uint8)
    decode_pack = audit(
        warmup(
            _pack_lhs_qsi8d32p_decode_kernel,
            (3,),
            x[:3],
            decode_lhs.view(torch.float16),
            decode_lhs.view(torch.int8),
            3,
            x.stride(0),
            K=k,
        )
    )
    check_common(decode_pack)
    assert decode_pack["frintn"] + decode_pack["fcvtns"] == 8
    assert decode_pack["umax"] == 3
    assert decode_pack["fmaxnm"] == 0
    assert decode_pack["fminnm"] == 0
    assert decode_pack["asm_lines"] < 400
    assert decode_pack["folded_spills"] == 0
    assert decode_pack["folded_reloads"] == 0

    matrix = {}
    stack_limits = {4: 0, 8: 6, 12: 11, 16: 8}
    for block_m in (4, 8, 12, 16):
        result = audit(
            warmup(
                _q4_prefill_i8mm_kai_kernel,
                (1, n // 4),
                lhs.view(torch.int8),
                lhs.view(torch.float16),
                rhs,
                rhs.view(torch.float16),
                out,
                N=n,
                K=k,
                BLOCK_M=block_m,
            )
        )
        check_common(result)
        assert result["smmla"] == 4 * block_m
        assert result["neon_smmla_intrinsics"] > 0
        assert result["stack_refs"] <= stack_limits[block_m]
        if groups > 8:
            assert result["annotated_inner_loops"] >= 1
        assert result["inner_loop_stack_refs"] == 0
        matrix[f"m{block_m}"] = result
    # M16 must not spill a live accumulator in the K32 loop.  Four folded
    # spills/reloads are the expected callee-saved d8-d15 prologue/epilogue;
    # the former hot q-register pair made this five and expanded to 482 lines.
    assert matrix["m16"]["asm_lines"] < 460
    assert matrix["m16"]["folded_spills"] == 4
    assert matrix["m16"]["folded_reloads"] == 4

    decode = audit(
        warmup(
            _q4_decode_sdot_kai_kernel,
            (1,),
            decode_lhs,
            rhs,
            out,
            0,
            n // 4,
            K=k,
            N=n,
            UNROLL=_decode_unroll(k),
        )
    )
    check_common(decode)
    assert decode["sdot"] >= 8 and decode["sdot"] % 8 == 0
    assert decode["addp"] >= 1
    assert decode["neon_sdot_intrinsics"] > 0
    return {
        "lhs_prefill_pack": panel4_pack,
        "lhs_decode_pack": decode_pack,
        "prefill_matrix": matrix,
        "decode_matrix": decode,
        "retired_row_lhs_pack": row_pack,
    }


def audit_q8_prefill_only(k: int, n: int) -> dict[str, object]:
    """Audit ordinary Q8 pack/tl.dot kernels without development TLE ops."""
    m = 16
    x = torch.randn((m, k), dtype=torch.bfloat16)
    weight_nk = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    weight_scale = torch.rand(n, dtype=torch.float32) / 127.0
    panel_stride = 4 * k + 16
    lhs_kai = torch.empty((m // 4) * panel_stride, dtype=torch.int8)
    rhs_kai = w8.pack_weights_i8mm_kai(weight_nk, weight_scale)
    out_prefill = torch.empty((m, n), dtype=torch.bfloat16)

    pack_prefill = audit(
        warmup(
            w8._pack_lhs_w8_i8mm_kai_kernel,
            (m,),
            x,
            lhs_kai,
            m,
            x.stride(0),
            K=k,
            FULL_ROWS=True,
        )
    )
    check_common(pack_prefill)
    assert pack_prefill["umax"] >= 3
    assert pack_prefill["umax"] % 3 == 0
    assert pack_prefill["fmaxnm"] == pack_prefill["umax"] // 3 + 1
    assert pack_prefill["fminnm"] == 0
    assert pack_prefill["asm_lines"] < 600
    assert pack_prefill["folded_spills"] == 0
    assert pack_prefill["folded_reloads"] == 0

    matrix = audit(
        warmup(
            w8._w8_prefill_i8mm_kai_kernel,
            (1, n // 4),
            lhs_kai,
            rhs_kai,
            out_prefill,
            N=n,
            K=k,
            BLOCK_M=16,
        )
    )
    check_common(matrix)
    assert matrix["smmla"] == 64
    check_q8_i8mm(matrix)
    assert matrix["asm_lines"] < 600
    assert matrix["folded_spills"] == 0
    assert matrix["folded_reloads"] == 0

    tails = {}
    for block_m in (4, 8):
        result = audit(
            warmup(
                w8._w8_prefill_i8mm_kai_short_tail_kernel,
                (1, n // 4),
                lhs_kai,
                rhs_kai,
                out_prefill,
                N=n,
                K=k,
                BLOCK_M=block_m,
            )
        )
        check_common(result)
        assert result["smmla"] == 4 * block_m
        check_q8_i8mm(result)
        assert result["folded_spills"] <= (0 if block_m == 4 else 2)
        assert result["folded_reloads"] <= (0 if block_m == 4 else 2)
        tails[f"m{block_m}"] = result

    m12 = audit(
        warmup(
            w8._w8_prefill_i8mm_kai_m12_kernel,
            (1, n // 4),
            lhs_kai,
            rhs_kai,
            out_prefill,
            N=n,
            K=k,
        )
    )
    check_common(m12)
    assert m12["smmla"] == 48
    check_q8_i8mm(m12)
    assert m12["folded_spills"] == 0
    assert m12["folded_reloads"] == 0
    tails["m12"] = m12
    return {
        "lhs_pack": pack_prefill,
        "matrix_m16": matrix,
        "tail_matrix": tails,
    }


def audit_q8(k: int, n: int) -> dict[str, object]:
    from flag_gems.runtime.backend._arm.ops.silu_and_mul import (
        _SWIGLU_TILE,
        _swiglu_ordinary_kernel,
    )

    x = torch.randn((16, k), dtype=torch.bfloat16)
    weight_nk = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    weight_kn = weight_nk.T.contiguous()
    weight_scale = torch.rand(n, dtype=torch.float32) / 127.0
    packed512 = w8.pack_weights_sdot_blocked(
        w8.pack_weights_sdot(weight_kn), 512
    )
    packed32 = w8.retile_weights_sdot_blocked(packed512, 512, 32)
    packed64 = w8.retile_weights_sdot_blocked(packed512, 512, 64)
    out_decode = torch.empty(n, dtype=torch.bfloat16)
    q_decode = torch.empty(k, dtype=torch.int8)
    x_scale_decode = torch.empty(1, dtype=torch.float32)
    quant_decode = audit(
        warmup(
            w8._quantize_bf16_w8_rne_kernel,
            (1,),
            x[0],
            q_decode,
            x_scale_decode,
            K=k,
            BLOCK_K=16,
        )
    )
    check_common(quant_decode)
    # LLVM may keep RNE as frintn+convert or select the equivalent fcvtns
    # directly.  Gate the number of vector RNE conversions, not one spelling.
    assert quant_decode["frintn"] + quant_decode["fcvtns"] == 4
    assert quant_decode["umax"] >= 3
    assert quant_decode["umax"] % 3 == 0
    assert quant_decode["fmaxnm"] == quant_decode["umax"] // 3 + 1
    assert quant_decode["fminnm"] == 0
    assert quant_decode["asm_lines"] < 400
    assert quant_decode["folded_spills"] == 0
    assert quant_decode["folded_reloads"] == 0

    whole = audit(
        warmup(
            w8._w8_decode_sdot_kernel,
            (1,),
            q_decode,
            x_scale_decode,
            packed64,
            weight_scale,
            out_decode,
            K=k,
            N=n,
            BLOCK_N=64,
            UNROLL=2,
            WHOLE_PROJECTION=True,
        )
    )
    check_common(whole)
    assert whole["sdot"] == 32
    assert whole["asm_lines"] < 400
    assert whole["folded_spills"] == 0
    assert whole["folded_reloads"] == 0

    vocabulary_n = 32768
    vocabulary_tile_n = w8.select_w8_decode_tile_n(
        vocabulary_n, 512
    )
    assert vocabulary_tile_n == 32
    vocabulary = audit(
        warmup(
            w8._w8_decode_sdot_kernel,
            (1,),
            q_decode,
            x_scale_decode,
            packed32,
            weight_scale,
            out_decode,
            K=k,
            N=vocabulary_n,
            BLOCK_N=vocabulary_tile_n,
            UNROLL=2,
            WHOLE_PROJECTION=True,
        )
    )
    check_common(vocabulary)
    assert vocabulary["sdot"] == 16
    assert vocabulary["asm_lines"] < 350
    assert vocabulary["folded_spills"] == 0
    assert vocabulary["folded_reloads"] == 0

    parallel = audit(
        warmup(
            w8._w8_decode_sdot_kernel,
            (n // 64,),
            q_decode,
            x_scale_decode,
            packed64,
            weight_scale,
            out_decode,
            K=k,
            N=n,
            BLOCK_N=64,
            UNROLL=2,
            WHOLE_PROJECTION=False,
        )
    )
    check_common(parallel)
    assert parallel["sdot"] == 32
    assert parallel["asm_lines"] < 400
    assert parallel["folded_spills"] == 0
    assert parallel["folded_reloads"] == 0

    m = 16
    panel_stride = 4 * k + 16
    lhs_kai = torch.empty((m // 4) * panel_stride, dtype=torch.int8)
    rhs_kai = w8.pack_weights_i8mm_kai(weight_nk, weight_scale)
    out_prefill = torch.empty((m, n), dtype=torch.bfloat16)
    pack_prefill = audit(
        warmup(
            w8._pack_lhs_w8_i8mm_kai_kernel,
            (m,),
            x,
            lhs_kai,
            m,
            x.stride(0),
            K=k,
            FULL_ROWS=True,
        )
    )
    check_common(pack_prefill)
    assert pack_prefill["umax"] >= 3
    assert pack_prefill["umax"] % 3 == 0
    assert pack_prefill["fmaxnm"] == pack_prefill["umax"] // 3 + 1
    assert pack_prefill["fminnm"] == 0
    assert pack_prefill["asm_lines"] < 600
    assert pack_prefill["folded_spills"] == 0
    assert pack_prefill["folded_reloads"] == 0

    matrix_prefill = audit(
        warmup(
            w8._w8_prefill_i8mm_kai_kernel,
            (1, n // 4),
            lhs_kai,
            rhs_kai,
            out_prefill,
            N=n,
            K=k,
            BLOCK_M=16,
        )
    )
    check_common(matrix_prefill)
    assert matrix_prefill["smmla"] == 64
    check_q8_i8mm(matrix_prefill)
    assert matrix_prefill["asm_lines"] < 600
    assert matrix_prefill["folded_spills"] == 0
    assert matrix_prefill["folded_reloads"] == 0

    tail_prefill = {}
    for block_m in (4, 8):
        result = audit(
            warmup(
                w8._w8_prefill_i8mm_kai_short_tail_kernel,
                (1, n // 4),
                lhs_kai,
                rhs_kai,
                out_prefill,
                N=n,
                K=k,
                BLOCK_M=block_m,
            )
        )
        check_common(result)
        assert result["smmla"] == 4 * block_m
        check_q8_i8mm(result)
        max_stack_refs = 0 if block_m == 4 else 2
        assert result["folded_spills"] <= max_stack_refs
        assert result["folded_reloads"] <= max_stack_refs
        assert result["asm_lines"] < (200 if block_m == 4 else 300)
        tail_prefill[f"m{block_m}"] = result

    m12_result = audit(
        warmup(
            w8._w8_prefill_i8mm_kai_m12_kernel,
            (1, n // 4),
            lhs_kai,
            rhs_kai,
            out_prefill,
            N=n,
            K=k,
        )
    )
    check_common(m12_result)
    assert m12_result["smmla"] == 48
    check_q8_i8mm(m12_result)
    assert m12_result["folded_spills"] == 0
    assert m12_result["folded_reloads"] == 0
    assert m12_result["asm_lines"] < 400
    tail_prefill["m12"] = m12_result

    qkv_n = n * 3
    qkv_weight_kn = weight_kn.repeat(1, 3)
    qkv_scale = weight_scale.repeat(3)
    qkv_pack = w8.pack_weights_sdot_blocked(
        w8.pack_weights_sdot(qkv_weight_kn), 64
    )
    qkv_out = torch.empty(qkv_n, dtype=torch.bfloat16)
    qkv_whole = audit(
        warmup(
            w8._w8_decode_sdot_kernel,
            (1,),
            q_decode,
            x_scale_decode,
            qkv_pack,
            qkv_scale,
            qkv_out,
            K=k,
            N=qkv_n,
            BLOCK_N=64,
            UNROLL=2,
            WHOLE_PROJECTION=True,
        )
    )
    check_common(qkv_whole)
    assert qkv_whole["sdot"] == 32
    assert qkv_whole["folded_spills"] == 0
    assert qkv_whole["folded_reloads"] == 0

    qkv_parallel = audit(
        warmup(
            w8._w8_decode_sdot_kernel,
            (qkv_n // 64,),
            q_decode,
            x_scale_decode,
            qkv_pack,
            qkv_scale,
            qkv_out,
            K=k,
            N=qkv_n,
            BLOCK_N=64,
            UNROLL=2,
            WHOLE_PROJECTION=False,
        )
    )
    check_common(qkv_parallel)
    assert qkv_parallel["sdot"] == 32
    assert qkv_parallel["folded_spills"] == 0
    assert qkv_parallel["folded_reloads"] == 0

    gate = torch.randn(n, dtype=torch.bfloat16)
    up = torch.randn(n, dtype=torch.bfloat16)
    mlp_out = torch.empty_like(gate)
    swiglu = audit(
        warmup(
            _swiglu_ordinary_kernel,
            (1,),
            gate,
            up,
            mlp_out,
            n,
            BLOCK_SIZE=_SWIGLU_TILE,
        )
    )
    check_common(swiglu)
    # The SLEEF-u10 polynomial is visible ordinary Triton arithmetic.  The
    # shape-specialized rolled loop must contain no external exp call or
    # duplicate dynamic tail body.
    assert swiglu["vector_math_calls"] == 0
    assert swiglu["folded_spills"] <= 4
    assert swiglu["folded_reloads"] <= 4
    assert swiglu["asm_lines"] < 600

    # Keep the former row-major prefill object visible as a regression record,
    # but do not compile legacy TLE-only decode entry points on the independent
    # Triton-CPU 3.7 port.
    q_legacy = torch.empty((m, k), dtype=torch.int8)
    scale_legacy = torch.empty(m, dtype=torch.float32)
    legacy_prefill = audit(
        warmup(
            w8._int8mm_dequant_bf16_kernel,
            (m // 8, n // 64),
            q_legacy,
            weight_kn,
            scale_legacy,
            weight_scale,
            out_prefill,
            m,
            n,
            k,
            q_legacy.stride(0),
            q_legacy.stride(1),
            weight_kn.stride(0),
            weight_kn.stride(1),
            out_prefill.stride(0),
            out_prefill.stride(1),
            BLOCK_M=8,
            BLOCK_N=64,
            BLOCK_K=32,
        )
    )
    return {
        "active_decode_single_core": whole,
        "active_decode_vocabulary_single_core": vocabulary,
        "active_decode_multicore_quant": quant_decode,
        "active_decode_multicore_matrix": parallel,
        "active_prefill_lhs_pack": pack_prefill,
        "active_prefill_matrix": matrix_prefill,
        "active_prefill_tail_matrix": tail_prefill,
        "active_qkv_single_core": qkv_whole,
        "active_qkv_multicore_matrix": qkv_parallel,
        "active_ordinary_swiglu_epilogue": swiglu,
        "retired_codegen": {
            "row_major_n64_prefill": legacy_prefill,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--q4-only", action="store_true")
    parser.add_argument("--q8-prefill-only", action="store_true")
    args = parser.parse_args()
    if args.q4_only and args.q8_prefill_only:
        raise ValueError("choose at most one audit subset")
    if args.k % 32 or args.n % 512:
        raise ValueError("audit shape requires K%32=0 and N%512=0")
    torch.set_num_threads(1)
    torch.manual_seed(9801)
    report: dict[str, object] = {
        "status": "PASS",
        "triton_module": triton.__file__,
        "target": "Arm CPU (feature-selected SVE2/Neon I8MM)",
        "shape": {"k": args.k, "n": args.n},
    }
    if args.q8_prefill_only:
        report["q8_prefill"] = audit_q8_prefill_only(args.k, args.n)
    else:
        report["q4"] = audit_q4(args.k, args.n)
    if not args.q4_only and not args.q8_prefill_only:
        report["q8"] = audit_q8(args.k, args.n)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
