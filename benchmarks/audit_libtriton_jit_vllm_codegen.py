#!/usr/bin/env python3
"""Audit the actual LLVM/AArch64 artifacts used by the vLLM Q4/Q8 router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


OPCODES = (
    "sdot",
    "smmla",
    "addp",
    "frintn",
    "fcvtns",
    "fcvtzs",
    "fcvtas",
)


def _instruction_count(assembly: str, opcode: str) -> int:
    count = 0
    for line in assembly.lower().splitlines():
        fields = line.lstrip().split(None, 1)
        count += bool(fields and fields[0] == opcode)
    return count


def _inner_loop_stack_refs(assembly: str) -> int:
    lines = assembly.lower().splitlines()
    references = 0
    for start, line in enumerate(lines):
        if "inner loop header" not in line:
            continue
        label = line.split(":", 1)[0].strip()
        if not label:
            continue
        for end in range(start + 1, len(lines)):
            branch = lines[end].strip()
            if branch.startswith("b") and label in branch:
                references += sum("[sp" in item for item in lines[start : end + 1])
                break
    return references


def _audit_file(path: Path) -> dict[str, object]:
    assembly = path.read_text().lower()
    llir_path = path.with_suffix(".llir")
    tttcir_path = path.with_suffix(".tttcir")
    if not llir_path.is_file() or not tttcir_path.is_file():
        raise FileNotFoundError(f"missing IR sibling for {path}")
    llir = llir_path.read_text().lower()
    tttcir = tttcir_path.read_text().lower()
    calls = [
        line.strip()
        for line in llir.splitlines()
        if " call " in line and "llvm." not in line and " asm " not in line
    ]
    result: dict[str, object] = {
        "cache_key": path.parent.name,
        "kernel": path.stem,
        "asm_lines": len(assembly.splitlines()),
        **{opcode: _instruction_count(assembly, opcode) for opcode in OPCODES},
        "stack_refs": sum("[sp" in line for line in assembly.splitlines()),
        "inner_loop_stack_refs": _inner_loop_stack_refs(assembly),
        "folded_spills": assembly.count("folded spill"),
        "folded_reloads": assembly.count("folded reload"),
        "external_calls": calls,
        "residual_triton_dot": "triton_cpu.dot" in tttcir,
        "sve_intrinsics": llir.count("llvm.aarch64.sve."),
        "sve_register_lines": sum(
            bool(
                re.search(
                    r"\b(?:z(?:[12]?\d|3[01])|p(?:[0-9]|1[0-5]))(?:\.[bhsd])?\b",
                    line,
                )
            )
            for line in assembly.splitlines()
        ),
    }
    return result


def _collect(cache: Path, names: set[str]) -> list[dict[str, object]]:
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    files = sorted(
        path for path in cache.rglob("*.asm") if path.stem in names
    )
    found = {path.stem for path in files}
    missing = names - found
    if missing:
        raise AssertionError(f"missing kernels in {cache}: {sorted(missing)}")
    return [_audit_file(path) for path in files]


def _collect_many(
    caches: list[Path], names: set[str]
) -> list[dict[str, object]]:
    files = sorted(
        path
        for cache in caches
        for path in cache.rglob("*.asm")
        if path.stem in names
    )
    missing = names - {path.stem for path in files}
    if missing:
        raise AssertionError(
            f"missing kernels in {[str(item) for item in caches]}: "
            f"{sorted(missing)}"
        )
    return [_audit_file(path) for path in files]


def _gate_common(items: list[dict[str, object]]) -> None:
    for item in items:
        assert item["external_calls"] == [], item
        assert not item["residual_triton_dot"], item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q4-cache", type=Path, required=True)
    parser.add_argument("--w8-cache", type=Path, required=True)
    parser.add_argument(
        "--w8-extra-cache",
        type=Path,
        action="append",
        default=[],
        help="additional exact-Q8 cache used to cover unobserved M tails",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    q4_names = {
        "_q4_fused_decode_asym_g32_kai_sdot_kernel",
        "_q4_fused_decode_asym_g128_kai_sdot_kernel",
        "_pack_lhs_qai8dxp_asym_panel4_kernel",
        "_q4_prefill_asym_i8mm_kai_kernel",
        "_q4_prefill_asym_g128_i8mm_kernel",
    }
    w8_names = {
        "_pack_lhs_qai8dxp_bf16_kernel",
        "_pack_lhs_qai8dxp_bf16_mr4_kernel",
        "_w8_qai8dxp_decode_sdot_kernel",
        "_w8_qai8dxp_prefill_short_tail_kernel",
        "_w8_qai8dxp_prefill_m12_kernel",
        "_w8_qai8dxp_prefill_i8mm_kernel",
    }
    q4 = _collect(args.q4_cache.resolve(), q4_names)
    w8_caches = [args.w8_cache.resolve()] + [
        item.resolve() for item in args.w8_extra_cache
    ]
    w8 = _collect_many(w8_caches, w8_names)
    _gate_common(q4)
    _gate_common(w8)

    q4_g128_decode = [
        item
        for item in q4
        if item["kernel"] == "_q4_fused_decode_asym_g128_kai_sdot_kernel"
    ]
    assert all(
        item["sdot"] == 32
        and item["addp"] == 4
        and item["fcvtas"] == 8
        for item in q4_g128_decode
    )
    q4_head_decode = [
        item
        for item in q4
        if item["kernel"] == "_q4_fused_decode_asym_g32_kai_sdot_kernel"
    ]
    assert all(
        int(item["sdot"]) in {8, 32}
        and int(item["addp"]) == int(item["sdot"]) // 8
        and item["fcvtas"] == 8
        and item["inner_loop_stack_refs"] == 0
        for item in q4_head_decode
    )
    q4_pack = [
        item
        for item in q4
        if item["kernel"] == "_pack_lhs_qai8dxp_asym_panel4_kernel"
    ]
    assert all(item["fcvtas"] == 32 for item in q4_pack)
    q4_prefill = [
        item for item in q4 if "_q4_prefill_asym_g128_i8mm" in str(item["kernel"])
    ]
    q4_smmla = {int(item["smmla"]) for item in q4_prefill}
    assert {32, 48, 64}.issubset(q4_smmla), q4_smmla
    assert all(int(item["smmla"]) > 0 for item in q4_prefill)
    q4_head_prefill = [
        item
        for item in q4
        if item["kernel"] == "_q4_prefill_asym_i8mm_kai_kernel"
    ]
    q4_head_smmla = {int(item["smmla"]) for item in q4_head_prefill}
    assert q4_head_smmla == {16, 32, 48}, q4_head_smmla
    # M16 was deliberately replaced by a paired-M8 grid after CIX A/B.  M4
    # is spill-free; the joined M8/M12 objects retain bounded spills because
    # they are still 9.7--13.4% faster than splitting every tile to M4.
    assert all(
        int(item["inner_loop_stack_refs"]) <= 11
        and int(item["stack_refs"]) <= 30
        and int(item["folded_spills"]) <= 10
        for item in q4_head_prefill
    )
    assert all(
        int(item["stack_refs"]) == 0
        for item in q4_head_prefill
        if int(item["smmla"]) == 16
    )
    # All other Q4 objects must stay free of stack traffic in an actual loop.
    # Function-level stack slots in the fully unrolled matrix body or a
    # callee-save prologue are reported separately.
    assert all(
        int(item["inner_loop_stack_refs"]) == 0
        for item in q4
        if item["kernel"] != "_q4_prefill_asym_i8mm_kai_kernel"
    )

    w8_pack_m1 = [
        item
        for item in w8
        if item["kernel"] == "_pack_lhs_qai8dxp_bf16_kernel"
    ]
    assert all(
        int(item["fcvtns"]) == 3
        and item["stack_refs"] == 0
        and item["inner_loop_stack_refs"] == 0
        for item in w8_pack_m1
    )
    w8_pack_mr4 = [
        item
        for item in w8
        if item["kernel"] == "_pack_lhs_qai8dxp_bf16_mr4_kernel"
    ]
    assert all(
        int(item["fcvtns"]) == 9
        and item["inner_loop_stack_refs"] == 0
        and int(item["folded_spills"]) <= 3
        for item in w8_pack_mr4
    )
    w8_decode = [
        item
        for item in w8
        if item["kernel"] == "_w8_qai8dxp_decode_sdot_kernel"
    ]
    assert all(
        item["sdot"] == 16
        and item["addp"] == 1
        and item["inner_loop_stack_refs"] == 0
        for item in w8_decode
    )
    w8_matrix = [
        item
        for item in w8
        if "_w8_qai8dxp_prefill" in str(item["kernel"])
    ]
    w8_smmla = {int(item["smmla"]) for item in w8_matrix}
    assert {16, 32, 48, 64}.issubset(w8_smmla), w8_smmla
    assert all(
        item["inner_loop_stack_refs"] == 0
        and int(item["folded_spills"]) <= 3
        for item in w8_matrix
    )

    result = {
        "status": "PASS",
        "q4_cache": str(args.q4_cache.resolve()),
        "w8_cache": str(args.w8_cache.resolve()),
        "w8_extra_caches": [str(item) for item in w8_caches[1:]],
        "q4": q4,
        "w8": w8,
        "summary": {
            "q4_specializations": len(q4),
            "w8_specializations": len(w8),
            "q4_g128_sdot_counts": sorted(
                {int(item["sdot"]) for item in q4_g128_decode}
            ),
            "q4_head_g32_sdot_counts": sorted(
                {int(item["sdot"]) for item in q4_head_decode}
            ),
            "q4_smmla_counts": sorted(q4_smmla),
            "q4_head_g32_smmla_counts": sorted(q4_head_smmla),
            "q4_decode_fcvtas_counts": sorted(
                {int(item["fcvtas"]) for item in q4_g128_decode}
            ),
            "q4_pack_fcvtas_counts": sorted(
                {int(item["fcvtas"]) for item in q4_pack}
            ),
            "w8_sdot_counts": sorted({int(item["sdot"]) for item in w8_decode}),
            "w8_smmla_counts": sorted(w8_smmla),
            "q4_hot_stack_specializations": sum(
                int(item["inner_loop_stack_refs"]) > 0 for item in q4
            ),
            "w8_hot_stack_specializations": sum(
                int(item["inner_loop_stack_refs"]) > 0 for item in w8
            ),
            "q4_stack_frame_specializations": sum(
                int(item["stack_refs"]) > 0 for item in q4
            ),
            "w8_stack_frame_specializations": sum(
                int(item["stack_refs"]) > 0 for item in w8
            ),
            "external_compute_calls": sum(bool(item["external_calls"]) for item in q4 + w8),
            "residual_triton_dot": sum(bool(item["residual_triton_dot"]) for item in q4 + w8),
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
