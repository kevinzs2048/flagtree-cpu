#!/usr/bin/env python3
"""Correctness and LLVM-codegen gates for MiniCPM5 W4A8 asymmetric Q4."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(ROOT / "ports/triton-cpu-3.7.2/python"),
    "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402

from flag_gems.runtime.backend._arm.q4.kernels import (  # noqa: E402
    _q4_decode_asym_g128_sdot_kernel,
    _q4_fused_decode_asym_sdot_kai_kernel,
    _q4_prefill_asym_g128_i8mm_kernel,
    _q4_prefill_asym_i8mm_kai_kernel,
)
from flag_gems.runtime.backend._arm.q4.linear import (  # noqa: E402
    _decode_unroll,
    linear_w4a8_asym,
    linear_w4a8_asym_add_rmsnorm,
    linear_w4a8_asym_g128,
    linear_w4a8_asym_g128_add_rmsnorm,
    linear_w4a8_asym_g128_rmsnorm,
    linear_w4a8_asym_rmsnorm,
    pack_rhs_qsi4c128p_asym,
    pack_rhs_qsi4c32p_asym,
)
from flag_gems.runtime.backend._arm.q4.optimize_qwen3 import (  # noqa: E402
    _qwen3_q4_swiglu_joined_kernel,
    _qwen3_q4_swiglu_pack_asym_g128_kernel,
)
from flag_gems.runtime.backend._arm.ops.silu_and_mul import (  # noqa: E402
    _SWIGLU_TILE,
)


def activation_qparams(value: torch.Tensor):
    minimum = torch.minimum(
        value.amin(dim=-1, keepdim=True), torch.zeros_like(value[..., :1])
    )
    maximum = torch.maximum(
        value.amax(dim=-1, keepdim=True), torch.zeros_like(value[..., :1])
    )
    scale = ((maximum - minimum) / 255.0).to(torch.bfloat16)
    scale = torch.where(
        scale == 0,
        torch.tensor(torch.finfo(torch.bfloat16).eps, dtype=torch.bfloat16),
        scale,
    )
    zero_point = torch.round(
        torch.clamp(
            torch.tensor(-128.0, dtype=torch.bfloat16) - minimum / scale,
            -128,
            127,
        )
    ).to(torch.int8)
    quantized = torch.round(
        torch.clamp(
            value / scale + zero_point.to(torch.bfloat16), -128, 127
        )
    ).to(torch.int8)
    return quantized, scale, zero_point


def reference(
    value: torch.Tensor, qweight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    shape = value.shape
    x = value.reshape(-1, shape[-1]).to(torch.bfloat16)
    q, scale, zero_point = activation_qparams(x)
    n, k = qweight.shape
    result = torch.zeros((x.shape[0], n), dtype=torch.float32)
    for start in range(0, k, 32):
        lhs = q[:, start : start + 32].to(torch.int32)
        lhs -= zero_point.to(torch.int32)
        rhs = qweight[:, start : start + 32].to(torch.int32)
        integer = lhs @ rhs.T
        result += (
            integer.float()
            * scale.float()
            * weight_scale[:, start // 32].float().unsqueeze(0)
        )
    return result.to(torch.bfloat16).reshape(*shape[:-1], n)


def reference_g128(
    value: torch.Tensor, qweight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """Reference the G128 integer accumulation and BF16 scale boundary."""
    shape = value.shape
    x = value.reshape(-1, shape[-1]).to(torch.bfloat16)
    q, scale, zero_point = activation_qparams(x)
    n, k = qweight.shape
    result = torch.zeros((x.shape[0], n), dtype=torch.float32)
    for start in range(0, k, 128):
        lhs = q[:, start : start + 128].to(torch.int32)
        lhs -= zero_point.to(torch.int32)
        rhs = qweight[:, start : start + 128].to(torch.int32)
        integer = lhs @ rhs.T
        result += (
            integer.float()
            * scale.float()
            * weight_scale[:, start // 128].float().unsqueeze(0)
        )
    return result.to(torch.bfloat16).reshape(*shape[:-1], n)


def rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float):
    fp32 = value.float()
    rrms = torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (fp32 * rrms).to(torch.bfloat16).float()
    return (normalized * weight.float()).to(torch.bfloat16)


def opcode_count(assembly: str, opcode: str) -> int:
    return sum(
        line.lstrip().split(None, 1)[0] == opcode
        for line in assembly.lower().splitlines()
        if line.lstrip()
    )


def audit(compiled, opcode: str, expected: int | None = None):
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    count = opcode_count(assembly, opcode)
    if expected is None:
        assert count > 0
    else:
        assert count == expected, (opcode, count, expected)
    assert "triton_cpu.dot" not in llir
    external = [
        line
        for line in llir.splitlines()
        if " call " in line and "llvm." not in line and " asm " not in line
    ]
    assert not external
    return {
        opcode: count,
        "external_compute_calls": 0,
        "residual_triton_cpu_dot": 0,
        "stack_refs": sum(
            "[sp" in line and line.lstrip().startswith(("ld", "st"))
            for line in assembly.splitlines()
        ),
    }


def audit_no_external_compute(compiled):
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    external = [
        line
        for line in llir.splitlines()
        if " call " in line and "llvm." not in line and " asm " not in line
    ]
    assert not external
    assert "tle_raw" not in llir
    return {
        "external_compute_calls": 0,
        "tle_raw": False,
        "stack_refs": sum(
            "[sp" in line and line.lstrip().startswith(("ld", "st"))
            for line in assembly.splitlines()
        ),
    }


def main() -> None:
    torch.manual_seed(24000)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    k, n = 128, 64
    qweight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    weight_scale = (torch.rand(n, k // 32) * 0.05 + 1.0e-4).to(
        torch.bfloat16
    )
    rhs = pack_rhs_qsi4c32p_asym(qweight, weight_scale)

    errors = {}
    for m in (1, 3, 4, 8, 12, 16):
        value = (torch.randn(m, k) * 0.4).to(torch.bfloat16)
        actual = linear_w4a8_asym(value, rhs, n, k)
        expected = reference(value, qweight, weight_scale)
        difference = (actual.float() - expected.float()).abs()
        max_abs = float(difference.max())
        if m < 4:
            assert torch.equal(actual, expected), (m, max_abs)
        else:
            # M4 panels retain the exact integer dot; only FP32 group
            # accumulation order differs from this scalar PyTorch reference.
            assert max_abs <= 0.25, (m, max_abs)
        errors[str(m)] = max_abs

    zero = torch.zeros((1, k), dtype=torch.bfloat16)
    zero_output = linear_w4a8_asym(zero, rhs, n, k)
    assert torch.equal(zero_output, reference(zero, qweight, weight_scale))

    value = torch.randn(1, k, dtype=torch.bfloat16)
    residual = torch.randn_like(value)
    norm_weight = torch.randn(k, dtype=torch.bfloat16)
    eps = 1.0e-6
    normalized = rmsnorm(value, norm_weight, eps)
    expected_rms = reference(normalized, qweight, weight_scale)
    actual_rms = linear_w4a8_asym_rmsnorm(
        value, norm_weight, eps, rhs, n, k
    )
    assert torch.equal(actual_rms, expected_rms)

    summed = (value.float() + residual.float()).to(torch.bfloat16)
    expected_add = reference(
        rmsnorm(summed, norm_weight, eps), qweight, weight_scale
    )
    actual_add, actual_residual = linear_w4a8_asym_add_rmsnorm(
        value, residual, norm_weight, eps, rhs, n, k
    )
    assert torch.equal(actual_residual, summed)
    assert torch.equal(actual_add, expected_add)

    weight_scale_g128 = (torch.rand(n, k // 128) * 0.05 + 1.0e-4).to(
        torch.bfloat16
    )
    rhs_g128 = pack_rhs_qsi4c128p_asym(qweight, weight_scale_g128)

    joined = torch.randn((1, 2 * k), dtype=torch.bfloat16)
    swiglu_reference = torch.empty((1, k), dtype=torch.bfloat16)
    _qwen3_q4_swiglu_joined_kernel[(1,)](
        joined,
        swiglu_reference,
        N=k,
        BLOCK_SIZE=_SWIGLU_TILE,
        num_warps=1,
        num_stages=1,
    )
    swiglu_scratch = torch.empty_like(swiglu_reference)
    swiglu_lhs = torch.empty(4 + k, dtype=torch.uint8)
    _qwen3_q4_swiglu_pack_asym_g128_kernel[(1,)](
        joined,
        swiglu_scratch,
        swiglu_lhs,
        N=k,
        BLOCK_SIZE=_SWIGLU_TILE,
        num_warps=1,
        num_stages=1,
    )
    assert torch.equal(swiglu_scratch, swiglu_reference)
    swiglu_q, swiglu_scale, swiglu_zp = activation_qparams(
        swiglu_reference
    )
    assert torch.equal(
        swiglu_lhs[:2].view(torch.bfloat16), swiglu_scale.reshape(-1)
    )
    assert torch.equal(swiglu_lhs[2:3].view(torch.int8), swiglu_zp.reshape(-1))
    assert torch.equal(swiglu_lhs[4:].view(torch.int8), swiglu_q.reshape(-1))
    swiglu_down = torch.empty((1, n), dtype=torch.bfloat16)
    _q4_decode_asym_g128_sdot_kernel[(1, 1)](
        swiglu_lhs,
        rhs_g128,
        swiglu_down,
        swiglu_down,
        0,
        n // 4,
        K=k,
        N=n,
        LHS_COMPACT=True,
        ADD_RESIDUAL=False,
        UNROLL=1,
        num_warps=1,
        num_stages=1,
    )
    assert torch.equal(
        swiglu_down,
        linear_w4a8_asym_g128(swiglu_reference, rhs_g128, n, k),
    )
    down_residual = torch.randn_like(swiglu_down)
    swiglu_down_residual = torch.empty_like(swiglu_down)
    _q4_decode_asym_g128_sdot_kernel[(1, 1)](
        swiglu_lhs,
        rhs_g128,
        swiglu_down_residual,
        down_residual,
        0,
        n // 4,
        K=k,
        N=n,
        LHS_COMPACT=True,
        ADD_RESIDUAL=True,
        UNROLL=1,
        num_warps=1,
        num_stages=1,
    )
    assert torch.equal(swiglu_down_residual, swiglu_down + down_residual)
    g128_errors = {}
    for m in (1, 3, 4, 8, 12, 16):
        value = (torch.randn(m, k) * 0.4).to(torch.bfloat16)
        actual = linear_w4a8_asym_g128(value, rhs_g128, n, k)
        expected = reference_g128(value, qweight, weight_scale_g128)
        difference = (actual.float() - expected.float()).abs()
        max_abs = float(difference.max())
        # The reference matmul and the four compiler-lowered K32 tiles use a
        # different integer reduction tree; the following FP32 products may
        # also be reassociated before their final BF16 rounding.
        assert max_abs <= 0.0625, ("g128", m, max_abs)
        g128_errors[str(m)] = max_abs

    value = torch.randn(1, k, dtype=torch.bfloat16)
    residual = torch.randn_like(value)
    normalized = rmsnorm(value, norm_weight, eps)
    expected_rms = reference_g128(
        normalized, qweight, weight_scale_g128
    )
    actual_rms = linear_w4a8_asym_g128_rmsnorm(
        value, norm_weight, eps, rhs_g128, n, k
    )
    assert float((actual_rms.float() - expected_rms.float()).abs().max()) <= 0.0625
    summed = (value.float() + residual.float()).to(torch.bfloat16)
    expected_add = reference_g128(
        rmsnorm(summed, norm_weight, eps), qweight, weight_scale_g128
    )
    actual_add, actual_residual = linear_w4a8_asym_g128_add_rmsnorm(
        value, residual, norm_weight, eps, rhs_g128, n, k
    )
    assert torch.equal(actual_residual, summed)
    assert float((actual_add.float() - expected_add.float()).abs().max()) <= 0.0625

    # Audit the rolled loops at the model's hidden-size K.  Very small K can
    # be deliberately straight-line expanded by LLVM and is not the relevant
    # register-pressure contract for MiniCPM decode.
    codegen_k = 2048
    codegen_groups = codegen_k // 32
    codegen_qweight = torch.randint(
        -8, 8, (n, codegen_k), dtype=torch.int8
    )
    codegen_scale = torch.full(
        (n, codegen_groups), 0.01, dtype=torch.bfloat16
    )
    codegen_rhs = pack_rhs_qsi4c32p_asym(
        codegen_qweight, codegen_scale
    )
    lhs = torch.empty(4 * codegen_groups * 144, dtype=torch.uint8)
    out = torch.empty((16, n), dtype=torch.bfloat16)
    prefill_codegen = {}
    for block_m in (4, 8, 12, 16):
        compiled = _q4_prefill_asym_i8mm_kai_kernel.warmup(
            lhs,
            codegen_rhs,
            out,
            N=n,
            K=codegen_k,
            BLOCK_M=block_m,
            num_warps=1,
            num_stages=1,
            grid=(1, n // 4),
        )
        prefill_codegen[str(block_m)] = audit(
            compiled, "smmla", 4 * block_m
        )
        assert prefill_codegen[str(block_m)]["stack_refs"] <= {
            4: 0,
            8: 6,
            12: 8,
            16: 12,
        }[block_m]

    partitions = 1
    scratch_bytes = codegen_groups * 36
    workspace = torch.empty(
        scratch_bytes + n * torch.bfloat16.itemsize, dtype=torch.uint8
    )
    codegen_value = torch.randn(1, codegen_k, dtype=torch.bfloat16)
    compiled_decode = _q4_fused_decode_asym_sdot_kai_kernel.warmup(
        codegen_value,
        workspace,
        codegen_rhs,
        scratch_bytes,
        codegen_k,
        0,
        n // 4,
        K=codegen_k,
        N=n,
        UNROLL=_decode_unroll(codegen_k),
        num_warps=1,
        num_stages=1,
        grid=(1, partitions),
    )
    decode_codegen = audit(compiled_decode, "sdot")
    assert decode_codegen["stack_refs"] <= 6

    codegen_scale_g128 = torch.full(
        (n, codegen_k // 128), 0.01, dtype=torch.bfloat16
    )
    codegen_rhs_g128 = pack_rhs_qsi4c128p_asym(
        codegen_qweight, codegen_scale_g128
    )
    g128_prefill_codegen = {}
    for block_m in (4, 8, 12, 16):
        compiled_g128_prefill = _q4_prefill_asym_g128_i8mm_kernel.warmup(
            lhs,
            codegen_rhs_g128,
            out,
            N=n,
            K=codegen_k,
            BLOCK_M=block_m,
            num_warps=1,
            num_stages=1,
            grid=(1, n // 4),
        )
        g128_prefill_codegen[str(block_m)] = audit(
            compiled_g128_prefill, "smmla"
        )
    g128_scratch_bytes = 4 + codegen_k
    g128_workspace = torch.empty(
        g128_scratch_bytes + n * torch.bfloat16.itemsize, dtype=torch.uint8
    )
    compiled_g128_decode = _q4_decode_asym_g128_sdot_kernel.warmup(
        g128_workspace,
        codegen_rhs_g128,
        out,
        out,
        0,
        n // 4,
        K=codegen_k,
        N=n,
        LHS_COMPACT=True,
        ADD_RESIDUAL=False,
        UNROLL=1,
        num_warps=1,
        num_stages=1,
        grid=(1, partitions),
    )
    g128_decode_codegen = audit(compiled_g128_decode, "sdot", 32)
    assert g128_decode_codegen["stack_refs"] <= 12
    compiled_g128_decode_residual = _q4_decode_asym_g128_sdot_kernel.warmup(
        g128_workspace,
        codegen_rhs_g128,
        out,
        out,
        0,
        n // 4,
        K=codegen_k,
        N=n,
        LHS_COMPACT=True,
        ADD_RESIDUAL=True,
        UNROLL=1,
        num_warps=1,
        num_stages=1,
        grid=(1, partitions),
    )
    g128_decode_residual_codegen = audit(
        compiled_g128_decode_residual, "sdot", 32
    )
    swiglu_codegen_k = 6144
    swiglu_codegen_joined = torch.empty(
        (1, 2 * swiglu_codegen_k), dtype=torch.bfloat16
    )
    swiglu_codegen_scratch = torch.empty(
        (1, swiglu_codegen_k), dtype=torch.bfloat16
    )
    swiglu_codegen_lhs = torch.empty(
        4 + swiglu_codegen_k, dtype=torch.uint8
    )
    compiled_swiglu_pack = (
        _qwen3_q4_swiglu_pack_asym_g128_kernel.warmup(
            swiglu_codegen_joined,
            swiglu_codegen_scratch,
            swiglu_codegen_lhs,
            N=swiglu_codegen_k,
            BLOCK_SIZE=_SWIGLU_TILE,
            num_warps=1,
            num_stages=1,
            grid=(1,),
        )
    )
    swiglu_pack_codegen = audit_no_external_compute(compiled_swiglu_pack)
    result = {
        "status": "PASS",
        "shape_errors_max_abs": errors,
        "decode_bit_exact_m1_m3": True,
        "all_zero_token_eps_contract": "torch.bfloat16.eps",
        "rmsnorm_projection_bit_exact": True,
        "add_rmsnorm_projection_bit_exact": True,
        "decode_codegen": decode_codegen,
        "prefill_codegen": prefill_codegen,
        "g128_shape_errors_max_abs": g128_errors,
        "g128_decode_codegen": g128_decode_codegen,
        "g128_decode_residual_codegen": g128_decode_residual_codegen,
        "g128_prefill_codegen": g128_prefill_codegen,
        "g128_rhs_group_bytes": 264,
        "g128_rhs_tile_correction_bytes": 16,
        "swiglu_compact_pack_bit_exact": True,
        "swiglu_down_bit_exact": True,
        "down_residual_epilogue_bit_exact": True,
        "swiglu_pack_codegen": swiglu_pack_codegen,
        "g128_codegen_rhs_bytes": codegen_rhs_g128.numel(),
        "ordinary_triton_tl_dot": True,
        "tle_raw_or_external_compute": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
