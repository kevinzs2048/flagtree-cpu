#!/usr/bin/env python3
"""Correctness, routing, Dynamo, and ISA checks for the ARM Q4 router."""

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
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")
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


def uses_fixed_i8mm() -> bool:
    if os.getenv("TRITON_CPU_FIXED_I8MM", "0").lower() in {
        "1",
        "true",
        "on",
    }:
        return True
    from triton._C.libtriton import llvm
    from triton.backends.cpu.target_info import (
        get_sve_vector_bits,
        supplement_aarch64_features,
    )

    features = supplement_aarch64_features(llvm.get_cpu_features())
    return not (
        "sve2" in features
        and "i8mm" in features
        and get_sve_vector_bits() == 128
    )

from flag_gems.runtime.backend._arm.q4.kernels import (  # noqa: E402
    _q4_fused_add_rmsnorm_decode_sdot_kai_kernel,
    _pack_lhs_qsi8d32p_decode_kernel,
    _pack_lhs_qsi8d32p_panel4_scalar_kernel,
    _pack_lhs_qsi8d32p_row_kernel,
    _q4_decode_sdot_kai_kernel,
    _q4_fused_decode_sdot_kai_kernel,
    _q4_fused_rmsnorm_decode_sdot_kai_kernel,
    _q4_fused_rmsnorm_qk_norm_decode_sdot_kai_kernel,
    _q4_prefill_i8mm_kai_kernel,
)
from flag_gems.runtime.backend._arm.q4.linear import (  # noqa: E402
    _decode_unroll,
    _use_m8_main_block,
    linear_w4a8,
    linear_w4a8_add_rmsnorm,
    linear_w4a8_rmsnorm,
    linear_w4a8_rmsnorm_qk_norm,
    pack_rhs_qsi4c32p,
    quantize_q4_0,
    set_fused_decode_enabled,
    stats,
)


ROUTE_SHAPES = (1, 2, 3, 4, 7, 8, 12, 15, 16, 17, 20, 24, 28, 31, 32, 47, 48)
MAX_BF16_ULP = 4
MAX_NEAR_ZERO_ABS = 2.0e-6


def expect_value_error(operation) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("operation did not reject an invalid Q4 input")


def check_public_input_validation(rhs: torch.Tensor, n: int, k: int) -> None:
    small_weight = torch.ones((4, 32), dtype=torch.bfloat16)
    nonfinite_weight = small_weight.clone()
    nonfinite_weight[0, 0] = float("nan")
    expect_value_error(lambda: quantize_q4_0(torch.ones((4, 32), dtype=torch.int8)))
    expect_value_error(lambda: quantize_q4_0(torch.empty((0, 32))))
    expect_value_error(lambda: quantize_q4_0(nonfinite_weight))

    qvalues = torch.zeros((4, 32), dtype=torch.int8)
    scales = torch.ones((4, 1), dtype=torch.float16)
    invalid_qvalues = qvalues.clone()
    invalid_qvalues[0, 0] = 8
    invalid_scales = scales.clone()
    invalid_scales[0, 0] = float("inf")
    expect_value_error(lambda: pack_rhs_qsi4c32p(invalid_qvalues, scales))
    expect_value_error(lambda: pack_rhs_qsi4c32p(qvalues, invalid_scales))

    valid_x = torch.ones((1, k), dtype=torch.bfloat16)
    noncontiguous_rhs = rhs.reshape(-1, 2).transpose(0, 1)
    expect_value_error(lambda: linear_w4a8(torch.tensor(1.0), rhs, n, k))
    expect_value_error(
        lambda: linear_w4a8(torch.empty((0, k), dtype=torch.bfloat16), rhs, n, k)
    )
    expect_value_error(
        lambda: linear_w4a8(torch.ones((1, k), dtype=torch.int8), rhs, n, k)
    )
    expect_value_error(lambda: linear_w4a8(valid_x, rhs.view(torch.int8), n, k))
    expect_value_error(lambda: linear_w4a8(valid_x, noncontiguous_rhs, n, k))
    expect_value_error(lambda: linear_w4a8(valid_x, rhs[:-1], n, k))
    expect_value_error(lambda: linear_w4a8(valid_x, rhs, 0, k))


def check_lhs_pack_finite_edges(k: int) -> None:
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
            0x3F00,
            0xBF00,
            0x4000,
            0xC000,
            0x7F7F,
            0xFF7F,
            0x3F81,
            0xBF81,
        ],
        dtype=torch.uint16,
    )
    repeats = (16 * k + edge_bits.numel() - 1) // edge_bits.numel()
    x = (
        edge_bits.repeat(repeats)[: 16 * k]
        .view(torch.bfloat16)
        .reshape(16, k)
    )
    groups = k // 32
    expected = torch.empty(4 * groups * 136, dtype=torch.uint8)
    actual = torch.empty_like(expected)
    _pack_lhs_qsi8d32p_row_kernel[(16,)](
        x,
        expected.view(torch.float16),
        expected.view(torch.int8),
        16,
        x.stride(0),
        K=k,
        num_warps=1,
        num_stages=1,
    )
    _pack_lhs_qsi8d32p_panel4_scalar_kernel[(4,)](
        x,
        actual.view(torch.float16),
        actual.view(torch.int8),
        16,
        x.stride(0),
        K=k,
        FULL_PANEL=True,
        num_warps=1,
        num_stages=1,
    )
    assert torch.equal(actual, expected)

    # The retired row kernel is an independent float-absmax/clamped reference.
    # Reassemble its KAI panel4 bytes into the compact decode ABI and require
    # the optimized integer-lane-max path to agree for every finite BF16 edge.
    for m in (1, 2, 3):
        panel = torch.zeros(groups * 136, dtype=torch.uint8)
        compact = torch.empty(m * groups * 34, dtype=torch.uint8)
        _pack_lhs_qsi8d32p_row_kernel[(m,)](
            x[:m],
            panel.view(torch.float16),
            panel.view(torch.int8),
            m,
            x.stride(0),
            K=k,
            num_warps=1,
            num_stages=1,
        )
        _pack_lhs_qsi8d32p_decode_kernel[(m,)](
            x[:m],
            compact.view(torch.float16),
            compact.view(torch.int8),
            m,
            x.stride(0),
            K=k,
            num_warps=1,
            num_stages=1,
        )
        reference = torch.empty_like(compact)
        for row in range(m):
            for group in range(groups):
                panel_base = group * 136
                compact_base = (row * groups + group) * 34
                reference[compact_base : compact_base + 2] = panel[
                    panel_base + row * 2 : panel_base + row * 2 + 2
                ]
                for segment in range(4):
                    source = panel_base + 8 + segment * 32 + row * 8
                    target = compact_base + 2 + segment * 8
                    reference[target : target + 8] = panel[
                        source : source + 8
                    ]
        assert torch.equal(compact, reference)


def reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    m, k = x.shape
    n = qweight.shape[0]
    groups = k // 32
    values = x.float().reshape(m, groups, 32)
    activation_scale = values.abs().amax(dim=-1) / 127.0
    inverse = torch.where(
        activation_scale != 0,
        1.0 / activation_scale,
        torch.zeros_like(activation_scale),
    )
    qactivation = (
        (values * inverse[..., None]).clamp(-127, 127).round().to(torch.int8)
    )
    activation_scale = activation_scale.to(torch.float16).float()
    result = torch.zeros((m, n), dtype=torch.float32)
    for group in range(groups):
        begin = group * 32
        dot = (
            qactivation[:, group].to(torch.int32)
            @ qweight[:, begin : begin + 32].to(torch.int32).T
        ).float()
        result += (
            dot
            * activation_scale[:, group, None]
            * weight_scale[:, group].float()[None, :]
        )
    return result.to(torch.bfloat16)


def bf16_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Return a monotonic BF16 encoding distance, including negative values."""
    actual_bits = actual.view(torch.int16).to(torch.int32) & 0xFFFF
    expected_bits = expected.view(torch.int16).to(torch.int32) & 0xFFFF

    def ordered(bits: torch.Tensor) -> torch.Tensor:
        magnitude = bits & 0x7FFF
        return torch.where(
            (bits & 0x8000) != 0,
            0x8000 - magnitude,
            0x8000 + magnitude,
        )

    return (ordered(actual_bits) - ordered(expected_bits)).abs()


def check_numerics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, int | float]:
    """Allow only sparse FP32 accumulation-order differences before BF16 store."""
    assert torch.isfinite(actual).all()
    distance = bf16_ulp_distance(actual, expected)
    absolute = (actual.float() - expected.float()).abs()
    mismatches = int((distance != 0).sum())
    max_ulp = int(distance.max())
    # Integer ULP distance is intentionally strict away from zero. Close to
    # zero, adjacent BF16 exponents can turn a one-micro-unit accumulation
    # difference into tens of encoding steps, so use a small absolute bound.
    bad = (distance > MAX_BF16_ULP) & (absolute > MAX_NEAR_ZERO_ABS)
    assert not bad.any(), (
        mismatches,
        max_ulp,
        float(absolute[bad].max()) if bad.any() else 0.0,
    )
    # A layout/packing error changes a broad region. Floating accumulation
    # ordering affects at most 0.026% of the tested K256-K6144 outputs at a
    # BF16 rounding boundary, so retain a conservative 1/2048 sparse bound.
    assert mismatches <= max(8, actual.numel() // 2048), mismatches
    return {
        "non_bit_exact": mismatches,
        "max_bf16_ulp": max_ulp,
        "max_abs_error": float(absolute.max()),
    }


def check_rhs_round_trip(
    rhs: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> None:
    n, k = qweight.shape
    groups = k // 32
    blocks = rhs.reshape(n // 4, groups, 72)
    got_scale = (
        blocks[:, :, :8]
        .contiguous()
        .view(torch.float16)
        .permute(0, 2, 1)
        .reshape(n, groups)
    )
    data = blocks[:, :, 8:].reshape(n // 4, groups, 2, 4, 8)
    packed = data.permute(0, 3, 1, 2, 4).reshape(n, groups, 2, 8)
    low = ((packed & 15).to(torch.int8) << 4) >> 4
    high = (packed & 0xF0).to(torch.int8) >> 4
    got_qweight = torch.cat(
        (low.reshape(n, groups, 16), high.reshape(n, groups, 16)), dim=-1
    ).reshape(n, k)
    assert torch.equal(got_scale, weight_scale)
    assert torch.equal(got_qweight, qweight)


def instruction_count(assembly: str, opcode: str) -> int:
    count = 0
    for line in assembly.lower().splitlines():
        fields = line.lstrip().split(None, 1)
        count += bool(fields and fields[0] == opcode)
    return count


def audit_compiled(
    compiled, opcode: str | tuple[str, ...], expected: int | None
) -> dict[str, int]:
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    opcodes = (opcode,) if isinstance(opcode, str) else opcode
    counts = {item: instruction_count(assembly, item) for item in opcodes}
    count = sum(counts.values())
    external_calls = sum(
        " call " in line and "llvm." not in line and " asm " not in line
        for line in llir.splitlines()
    )
    stack_lines = [
        index
        for index, line in enumerate(assembly.splitlines(), start=1)
        if "[sp" in line and line.lstrip().startswith(("ld", "st"))
    ]
    stack = len(stack_lines)
    if expected is None:
        assert count >= 8 and count % 8 == 0, (opcodes, counts)
    else:
        assert count == expected, (opcodes, counts, expected)
    assert "triton_cpu.dot" not in llir
    assert external_calls == 0
    return {
        **counts,
        "instruction_total": count,
        "external_calls": external_calls,
        "stack_refs": stack,
        "last_stack_ref_line": max(stack_lines, default=0),
    }


def check_codegen(k: int, n: int) -> dict[str, object]:
    groups = k // 32
    lhs = torch.empty(4 * groups * 136, dtype=torch.uint8)
    rhs = torch.empty((n // 4) * groups * 72, dtype=torch.uint8)
    output = torch.empty((16, n), dtype=torch.bfloat16)
    variants = {}
    pack_input = torch.empty((16, k), dtype=torch.bfloat16)
    pack = _pack_lhs_qsi8d32p_panel4_scalar_kernel.warmup(
        pack_input,
        lhs.view(torch.float16),
        lhs.view(torch.int8),
        16,
        pack_input.stride(0),
        K=k,
        FULL_PANEL=True,
        num_warps=1,
        num_stages=1,
        grid=(4,),
    )
    variants["lhs_pack"] = audit_compiled(pack, ("frintn", "fcvtns"), 32)
    pack_assembly = pack.asm["asm"].lower()
    variants["lhs_pack"]["umax"] = instruction_count(pack_assembly, "umax")
    variants["lhs_pack"]["fmaxnm"] = instruction_count(
        pack_assembly, "fmaxnm"
    )
    variants["lhs_pack"]["fminnm"] = instruction_count(
        pack_assembly, "fminnm"
    )
    assert variants["lhs_pack"]["umax"] == 12
    assert variants["lhs_pack"]["fmaxnm"] == 0
    assert variants["lhs_pack"]["fminnm"] == 0
    assert variants["lhs_pack"]["stack_refs"] == 0
    for block_m in (4, 8, 12, 16):
        compiled = _q4_prefill_i8mm_kai_kernel.warmup(
            lhs.view(torch.int8),
            lhs.view(torch.float16),
            rhs,
            rhs.view(torch.float16),
            output,
            N=n,
            K=k,
            BLOCK_M=block_m,
            num_warps=1,
            num_stages=1,
            grid=(1, n // 4),
        )
        variants[f"m{block_m}"] = audit_compiled(
            compiled, "smmla", 4 * block_m
        )
        assembly = compiled.asm["asm"].lower()
        variants[f"m{block_m}"]["vector_moves"] = sum(
            line.lstrip().startswith("mov\tv")
            for line in assembly.splitlines()
        )
    # Fixed-width M16 uses reverse panel order while SVE uses natural order.
    # Both schedules must avoid hot-loop accumulator rotation and retain only
    # the four callee-saved SIMD pairs in the prologue/epilogue.
    stack_limits = {4: 0, 8: 6, 12: 11, 16: 8}
    for block_m, limit in stack_limits.items():
        assert variants[f"m{block_m}"]["stack_refs"] <= limit
    assert variants["m16"]["vector_moves"] == 0

    decode_lhs = torch.empty(3 * groups * 34, dtype=torch.uint8)
    decode_pack = _pack_lhs_qsi8d32p_decode_kernel.warmup(
        pack_input[:3],
        decode_lhs.view(torch.float16),
        decode_lhs.view(torch.int8),
        3,
        pack_input.stride(0),
        K=k,
        num_warps=1,
        num_stages=1,
        grid=(3,),
    )
    variants["lhs_decode_pack"] = audit_compiled(
        decode_pack, ("frintn", "fcvtns"), 8
    )
    decode_pack_assembly = decode_pack.asm["asm"].lower()
    variants["lhs_decode_pack"]["umax"] = instruction_count(
        decode_pack_assembly, "umax"
    )
    variants["lhs_decode_pack"]["fmaxnm"] = instruction_count(
        decode_pack_assembly, "fmaxnm"
    )
    variants["lhs_decode_pack"]["fminnm"] = instruction_count(
        decode_pack_assembly, "fminnm"
    )
    assert variants["lhs_decode_pack"]["umax"] == 3
    assert variants["lhs_decode_pack"]["fmaxnm"] == 0
    assert variants["lhs_decode_pack"]["fminnm"] == 0
    assert variants["lhs_decode_pack"]["stack_refs"] == 0
    decode = _q4_decode_sdot_kai_kernel.warmup(
        decode_lhs,
        rhs,
        output,
        0,
        n // 4,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        num_warps=1,
        num_stages=1,
        grid=(1,),
    )
    variants["decode"] = audit_compiled(decode, "sdot", None)
    variants["decode"]["addp"] = instruction_count(
        decode.asm["asm"], "addp"
    )
    assert variants["decode"]["addp"] >= 1
    fused_partitions = 8
    fused_scratch_bytes = fused_partitions * groups * 34
    fused_workspace = torch.empty(
        fused_scratch_bytes + output[:1].numel() * output.element_size(),
        dtype=torch.uint8,
    )
    fused_decode = _q4_fused_decode_sdot_kai_kernel.warmup(
        pack_input[:1],
        fused_workspace,
        rhs,
        fused_scratch_bytes,
        pack_input.stride(0),
        0,
        n // 4,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        num_warps=1,
        num_stages=1,
        grid=(1, fused_partitions),
    )
    variants["fused_decode"] = audit_compiled(
        fused_decode,
        ("fcvtns", "sdot"),
        8 + 8 * _decode_unroll(k),
    )
    fused_assembly = fused_decode.asm["asm"].lower()
    variants["fused_decode"]["umax"] = instruction_count(
        fused_assembly, "umax"
    )
    variants["fused_decode"]["addp"] = instruction_count(
        fused_assembly, "addp"
    )
    assert variants["fused_decode"]["umax"] == 3
    assert variants["fused_decode"]["addp"] >= 1
    assert variants["fused_decode"]["stack_refs"] <= 2
    rms_weight = torch.randn(k, dtype=torch.bfloat16)
    fused_rmsnorm_decode = _q4_fused_rmsnorm_decode_sdot_kai_kernel.warmup(
        pack_input[:1],
        rms_weight,
        fused_workspace,
        rhs,
        fused_scratch_bytes,
        pack_input.stride(0),
        0,
        n // 4,
        1.0e-6,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        NORM_TILE=16,
        num_warps=1,
        num_stages=1,
        grid=(1, fused_partitions),
    )
    variants["fused_rmsnorm_decode"] = audit_compiled(
        fused_rmsnorm_decode,
        ("fcvtns", "sdot", "fsqrt"),
        8 + 8 * _decode_unroll(k) + 1,
    )
    fused_rmsnorm_assembly = fused_rmsnorm_decode.asm["asm"].lower()
    variants["fused_rmsnorm_decode"]["umax"] = instruction_count(
        fused_rmsnorm_assembly, "umax"
    )
    variants["fused_rmsnorm_decode"]["addp"] = instruction_count(
        fused_rmsnorm_assembly, "addp"
    )
    assert variants["fused_rmsnorm_decode"]["umax"] == 3
    assert variants["fused_rmsnorm_decode"]["addp"] >= 1
    qk_weight = torch.randn(32, dtype=torch.bfloat16)
    fused_rmsnorm_qk_norm_decode = (
        _q4_fused_rmsnorm_qk_norm_decode_sdot_kai_kernel.warmup(
            pack_input[:1],
            rms_weight,
            qk_weight,
            fused_workspace,
            rhs,
            fused_scratch_bytes,
            pack_input.stride(0),
            0,
            n // 4,
            1.0e-6,
            1.0e-6,
            K=k,
            N=n,
            UNROLL=_decode_unroll(k),
            Q_ELEMENTS=16,
            K_ELEMENTS=16,
            HEAD_DIM=16,
            NORM_TILE=16,
            num_warps=1,
            num_stages=1,
            grid=(1, 1),
        )
    )
    variants["fused_rmsnorm_qk_norm_decode"] = audit_compiled(
        fused_rmsnorm_qk_norm_decode,
        ("fcvtns", "sdot", "fsqrt"),
        8 + 8 * _decode_unroll(k) + 2,
    )
    rmsnorm_qk_assembly = fused_rmsnorm_qk_norm_decode.asm["asm"].lower()
    variants["fused_rmsnorm_qk_norm_decode"]["umax"] = instruction_count(
        rmsnorm_qk_assembly, "umax"
    )
    variants["fused_rmsnorm_qk_norm_decode"]["addp"] = instruction_count(
        rmsnorm_qk_assembly, "addp"
    )
    assert variants["fused_rmsnorm_qk_norm_decode"]["umax"] == 3
    assert variants["fused_rmsnorm_qk_norm_decode"]["addp"] >= 1
    summed_byte_offset = fused_scratch_bytes
    summed_bytes = fused_partitions * k * torch.bfloat16.itemsize
    add_rmsnorm_output_offset = summed_byte_offset + summed_bytes
    add_rmsnorm_workspace = torch.empty(
        add_rmsnorm_output_offset
        + output[:1].numel() * output.element_size(),
        dtype=torch.uint8,
    )
    fused_add_rmsnorm_decode = (
        _q4_fused_add_rmsnorm_decode_sdot_kai_kernel.warmup(
            pack_input[:1],
            pack_input[1:2],
            rms_weight,
            add_rmsnorm_workspace,
            rhs,
            summed_byte_offset,
            add_rmsnorm_output_offset,
            pack_input.stride(0),
            0,
            n // 4,
            1.0e-6,
            K=k,
            N=n,
            UNROLL=_decode_unroll(k),
            NORM_TILE=16,
            num_warps=1,
            num_stages=1,
            grid=(1, fused_partitions),
        )
    )
    variants["fused_add_rmsnorm_decode"] = audit_compiled(
        fused_add_rmsnorm_decode,
        ("fcvtns", "sdot", "fsqrt"),
        8 + 8 * _decode_unroll(k) + 1,
    )
    add_rmsnorm_assembly = fused_add_rmsnorm_decode.asm["asm"].lower()
    variants["fused_add_rmsnorm_decode"]["umax"] = instruction_count(
        add_rmsnorm_assembly, "umax"
    )
    variants["fused_add_rmsnorm_decode"]["addp"] = instruction_count(
        add_rmsnorm_assembly, "addp"
    )
    assert variants["fused_add_rmsnorm_decode"]["umax"] == 3
    assert variants["fused_add_rmsnorm_decode"]["addp"] >= 1
    # Six incoming scalar ABI arguments exceed AArch64's x0-x7 register
    # window.  Their loads are confined to the entry block, not K-loop spills.
    assert variants["fused_add_rmsnorm_decode"]["stack_refs"] == 6
    assert variants["fused_add_rmsnorm_decode"]["last_stack_ref_line"] <= 24
    return variants


def check_dynamic_graph(rhs: torch.Tensor, n: int, k: int) -> list[int]:
    def operation(x):
        return torch.ops.flag_gems.arm_q4_linear(x, rhs, n, k)

    compiled = torch.compile(operation, fullgraph=True, dynamic=True)
    shapes = (20, 1, 7, 32, 2, 12, 47, 3)
    for m in shapes:
        torch.manual_seed(9000 + m)
        x = torch.randn((m, k), dtype=torch.bfloat16)
        assert torch.equal(compiled(x), operation(x))
    return list(shapes)


def check_decode_route_equivalence(
    rhs: torch.Tensor, n: int, k: int
) -> None:
    """The production fusion must preserve the legacy route bit-for-bit."""
    previous = set_fused_decode_enabled(False)
    try:
        for m in (1, 2, 3):
            torch.manual_seed(7100 + m)
            x = torch.randn((m, k), dtype=torch.bfloat16)
            set_fused_decode_enabled(False)
            legacy = linear_w4a8(x, rhs, n, k)
            set_fused_decode_enabled(True)
            fused = linear_w4a8(x, rhs, n, k)
            assert torch.equal(fused, legacy)
    finally:
        set_fused_decode_enabled(previous)


def check_fused_rmsnorm_decode(
    rhs: torch.Tensor, n: int, k: int
) -> None:
    torch.manual_seed(8191)
    weight = torch.randn((k,), dtype=torch.bfloat16)
    eps = 1.0e-6
    for m in (1, 2, 3):
        x = torch.randn((m, k), dtype=torch.bfloat16)
        x_fp32 = x.float()
        rrms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
        normalized = (x_fp32 * rrms).to(torch.bfloat16)
        normalized = normalized * weight
        expected = linear_w4a8(normalized, rhs, n, k)
        actual = linear_w4a8_rmsnorm(x, weight, eps, rhs, n, k)
        assert torch.equal(actual, expected)


def check_fused_rmsnorm_qk_norm_decode(
    rhs: torch.Tensor, n: int, k: int
) -> None:
    # The small production gate uses D=16 so it remains valid for --n 64.
    # Full Qwen tests exercise D=128 and the large parallel partition shape.
    head_dim = 16
    q_elements = 16
    k_elements = 16
    torch.manual_seed(10103)
    input_weight = torch.randn((k,), dtype=torch.bfloat16)
    q_weight = torch.randn((head_dim,), dtype=torch.bfloat16)
    k_weight = torch.randn((head_dim,), dtype=torch.bfloat16)
    qk_weight = torch.cat((q_weight, k_weight)).contiguous()
    eps = 1.0e-6
    for m in (1, 2, 3):
        x = torch.randn((m, k), dtype=torch.bfloat16)
        x_fp32 = x.float()
        input_rrms = torch.rsqrt(
            x_fp32.square().mean(dim=-1, keepdim=True) + eps
        )
        normalized = (x_fp32 * input_rrms).to(torch.bfloat16)
        normalized = normalized * input_weight
        expected = linear_w4a8(normalized, rhs, n, k)
        for offset, weight in (
            (0, q_weight),
            (q_elements, k_weight),
        ):
            head = expected[:, offset : offset + head_dim].float()
            rrms = torch.rsqrt(
                head.square().mean(dim=-1, keepdim=True) + eps
            )
            expected[:, offset : offset + head_dim] = (
                (head * rrms).to(torch.bfloat16) * weight
            )
        actual = linear_w4a8_rmsnorm_qk_norm(
            x,
            input_weight,
            eps,
            qk_weight,
            eps,
            q_elements,
            k_elements,
            head_dim,
            rhs,
            n,
            k,
        )
        assert torch.equal(actual, expected)


def check_fused_add_rmsnorm_decode(
    rhs: torch.Tensor, n: int, k: int
) -> None:
    torch.manual_seed(12289)
    weight = torch.randn((k,), dtype=torch.bfloat16)
    eps = 1.0e-6
    for m in (1, 2, 3):
        x = torch.randn((m, k), dtype=torch.bfloat16)
        residual = torch.randn((m, k), dtype=torch.bfloat16)
        summed = (x.float() + residual.float()).to(torch.bfloat16)
        summed_fp32 = summed.float()
        rrms = torch.rsqrt(
            summed_fp32.square().mean(dim=-1, keepdim=True) + eps
        )
        normalized = (summed_fp32 * rrms).to(torch.bfloat16)
        normalized = normalized * weight
        expected = linear_w4a8(normalized, rhs, n, k)
        actual, updated_residual = linear_w4a8_add_rmsnorm(
            x, residual.clone(), weight, eps, rhs, n, k
        )
        assert torch.equal(actual, expected)
        assert torch.equal(updated_residual, summed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--skip-dynamo", action="store_true")
    args = parser.parse_args()
    if args.k % 32 or args.n % 4:
        raise ValueError("requires K%32=0 and N%4=0")

    torch.set_num_threads(1)
    torch.manual_seed(4817)
    weight = torch.randn((args.n, args.k), dtype=torch.bfloat16)
    qweight, weight_scale = quantize_q4_0(weight)
    rhs = pack_rhs_qsi4c32p(qweight, weight_scale)
    check_rhs_round_trip(rhs, qweight, weight_scale)
    check_public_input_validation(rhs, args.n, args.k)
    check_lhs_pack_finite_edges(args.k)
    check_decode_route_equivalence(rhs, args.n, args.k)
    check_fused_rmsnorm_decode(rhs, args.n, args.k)
    if args.n >= 64:
        check_fused_rmsnorm_qk_norm_decode(rhs, args.n, args.k)
    check_fused_add_rmsnorm_decode(rhs, args.n, args.k)

    checked = {}
    for m in ROUTE_SHAPES:
        torch.manual_seed(1000 + m)
        x = torch.randn((m, args.k), dtype=torch.bfloat16)
        actual = linear_w4a8(x, rhs, args.n, args.k)
        expected = reference(x, qweight, weight_scale)
        checked[str(m)] = check_numerics(actual, expected)

    codegen = check_codegen(args.k, args.n)
    dynamic_shapes = (
        [] if args.skip_dynamo else check_dynamic_graph(rhs, args.n, args.k)
    )
    router_stats = stats()
    if _use_m8_main_block():
        assert router_stats["m8_split_main_launches"] > 0
        assert router_stats["m16_main_launches"] == 0
    else:
        assert router_stats["m16_main_launches"] > 0
        assert router_stats["m8_split_main_launches"] == 0
    print(
        json.dumps(
            {
                "status": "PASS",
                "triton_module": triton.__file__,
                "k": args.k,
                "n": args.n,
                "rhs_round_trip": True,
                "public_input_validation": True,
                "lhs_pack_finite_edges": True,
                "decode_routes_bit_exact": True,
                "fused_rmsnorm_decode_bit_exact": True,
                "fused_rmsnorm_qk_norm_decode_bit_exact": args.n >= 64,
                "fused_add_rmsnorm_decode_bit_exact": True,
                "numeric_checks_by_m": checked,
                "max_allowed_bf16_ulp": MAX_BF16_ULP,
                "max_near_zero_abs": MAX_NEAR_ZERO_ABS,
                "dynamic_graph_m": dynamic_shapes,
                "codegen": codegen,
                "router_stats": router_stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
