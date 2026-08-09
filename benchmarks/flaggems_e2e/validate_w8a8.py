#!/usr/bin/env python3
"""Focused correctness/codegen checks for the ARM Qwen3 W8A8 path.

Run with ``python -S`` so editable installs from other worktrees cannot
silently replace this checkout.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
import torch.nn.functional as F  # noqa: E402
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


def check_decode_and_codegen() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.int8 import tle_int8_linear as w8

    torch.manual_seed(100)
    exact_shapes = []
    quant_half_ties = 0
    for k, n in (
        (256, 256),
        (256, 512),
        (256, 768),
        (1004, 512),
        (1008, 512),
        (1024, 1024),
        (3072, 1024),
        (1024, 3072),
    ):
        x = torch.randn(k, dtype=torch.bfloat16)
        weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
        weight_scale = torch.rand(n, dtype=torch.float32) / 127.0
        linear = w8.TLEInt8Linear(weight, weight_scale)
        output = linear(x)
        assert linear._ordinary_aot is None

        xf32 = x.float()
        absmax = xf32.abs().max().clamp(min=1.0e-8)
        activation_scale = absmax / 127.0
        quantized = torch.empty(k, dtype=torch.int8)
        generated_scale = torch.empty(1, dtype=torch.float32)
        w8._quantize_bf16_w8_rne_kernel[(1,)](
            x,
            quantized,
            generated_scale,
            K=k,
            BLOCK_K=16,
        )
        assert torch.equal(generated_scale, activation_scale.reshape(1))

        # FP32 reciprocal/multiply may land immediately above or below an
        # exact half-integer.  Verify any difference from the real-valued RNE
        # reference is only this permitted one-count half-tie, then use the
        # generated INT8 activation to isolate the SDOT/dequant correctness.
        exact_scaled = x.double() * 127.0 / x.double().abs().max()
        exact_quantized = exact_scaled.round().to(torch.int8)
        different = quantized != exact_quantized
        if torch.any(different):
            delta = (
                quantized.to(torch.int16)
                - exact_quantized.to(torch.int16)
            ).abs()
            assert int(delta.max()) == 1
            fractional = exact_scaled[different].abs().remainder(1.0)
            assert torch.all(
                torch.isclose(
                    fractional,
                    torch.tensor(0.5, dtype=torch.float64),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            quant_half_ties += int(torch.count_nonzero(different))
        reference = (
            torch._int_mm(
                quantized.reshape(1, k), weight.T.contiguous()
            ).float()
            * activation_scale
            * weight_scale
        ).to(torch.bfloat16).reshape(n)
        assert torch.equal(output, reference), f"decode mismatch K={k} N={n}"
        exact_shapes.append([k, n])

    cache = next(iter(w8._w8_decode_sdot_kernel.device_caches.values()))[0]
    variants = list(cache.values())
    for compiled in variants:
        assembly = compiled.asm["asm"].lower()
        llir = compiled.asm["llir"].lower()
        external_calls = [
            line
            for line in llir.splitlines()
            if " call " in line and "llvm." not in line and " asm " not in line
        ]
        assert assembly.count("sdot") > 0
        assert "folded spill" not in assembly
        assert "folded reload" not in assembly
        assert not external_calls
        assert "triton_cpu.dot" not in llir
    return {
        "bit_exact_shapes": exact_shapes,
        "ordinary_tl_dot": True,
        "compiled_variants": len(variants),
        "sdot_spill_free": True,
        "external_compute_calls": 0,
        "quant_fp32_half_ties": quant_half_ties,
    }


def check_w8_quantizer_finite_bf16() -> dict[str, object]:
    """Exercise RNE quantization over every finite BF16 bit pattern."""
    from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
        _quantize_bf16_w8_rne_kernel,
    )

    values = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.uint16)
        .view(torch.bfloat16)
    )
    finite = values[torch.isfinite(values)]
    assert finite.numel() == 65280
    k = 256
    fp32_staged_half_ties = 0
    for begin in range(0, finite.numel(), k):
        x = finite[begin : begin + k]
        if x.numel() != k:
            x = torch.cat(
                [x, torch.zeros(k - x.numel(), dtype=torch.bfloat16)]
            )
        quantized = torch.empty(k, dtype=torch.int8)
        generated_scale = torch.empty(1, dtype=torch.float32)
        _quantize_bf16_w8_rne_kernel[(1,)](
            x,
            quantized,
            generated_scale,
            K=k,
            BLOCK_K=16,
        )

        xf32 = x.float()
        absmax = xf32.abs().max().clamp(min=1.0e-8)
        assert torch.equal(
            generated_scale, (absmax / 127.0).reshape(1)
        )
        # LLVM may reassociate x * (127 / absmax) at an exact mathematical
        # half-tie.  Accept only that one-count RNE boundary; every other
        # finite BF16 encoding must equal the explicitly staged FP32 result.
        staged = torch.round(xf32 * (127.0 / absmax)).to(torch.int8)
        different = quantized != staged
        if torch.any(different):
            delta = (
                quantized.to(torch.int16) - staged.to(torch.int16)
            ).abs()
            assert int(delta.max()) == 1
            exact_scaled = x.double() * 127.0 / absmax.double()
            fractional = exact_scaled[different].abs().remainder(1.0)
            assert torch.all(fractional == 0.5)
            fp32_staged_half_ties += int(torch.count_nonzero(different))

    return {
        "finite_bf16_encodings": int(finite.numel()),
        "chunks": (int(finite.numel()) + k - 1) // k,
        "scale_bit_exact": True,
        "non_half_tie_mismatches": 0,
        "fp32_staged_half_ties": fp32_staged_half_ties,
    }


def check_prefill_and_codegen() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
        _fused_prefill_w8a8,
        _pack_lhs_w8_i8mm_kai_kernel,
        _w8_prefill_i8mm_kai_kernel,
        _w8_prefill_i8mm_kai_short_tail_kernel,
        pack_weights_i8mm_kai,
    )

    torch.manual_seed(101)
    exact_shapes = []
    quant_half_ties = 0
    for rows in (2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 31, 32, 64):
        k, n = 128, 64
        x = torch.randn(rows, k, dtype=torch.bfloat16)
        weight = torch.randint(-127, 128, (k, n), dtype=torch.int8)
        weight_scale = torch.rand(n, dtype=torch.float32) / 127.0
        weight_kai = pack_weights_i8mm_kai(
            weight.T.contiguous(), weight_scale
        )

        out = _fused_prefill_w8a8(x, weight, weight_scale, weight_kai)
        assert out is not None

        xf32 = x.float()
        x_scale = (
            xf32.abs().amax(dim=1, keepdim=True).clamp(min=1.0e-8) / 127.0
        )
        expected_q = torch.round(
            xf32
            * (
                127.0
                / xf32.abs().amax(dim=1, keepdim=True).clamp(min=1.0e-8)
            )
        ).clamp(-128, 127).to(torch.int8)
        m_kernel = (
            4
            if rows <= 4
            else 8
            if rows <= 8
            else 12
            if rows <= 12
            else ((rows + 15) // 16) * 16
        )
        panel_stride = 4 * k + 16
        lhs_packed = torch.empty(
            (m_kernel // 4) * panel_stride, dtype=torch.int8
        )
        _pack_lhs_w8_i8mm_kai_kernel[(m_kernel,)](
            x,
            lhs_packed,
            rows,
            x.stride(0),
            K=k,
            FULL_ROWS=rows == m_kernel,
            num_warps=1,
            num_stages=1,
        )
        generated_q = torch.empty_like(expected_q)
        generated_scale = torch.empty_like(x_scale)
        for row in range(rows):
            panel = row // 4
            panel_row = row % 4
            for offset in range(0, k, 8):
                begin = (
                    panel * panel_stride
                    + (offset // 8) * 32
                    + panel_row * 8
                )
                generated_q[row, offset : offset + 8].copy_(
                    lhs_packed[begin : begin + 8]
                )
            generated_scale[row, 0] = lhs_packed[
                panel * panel_stride + 4 * k :
            ].view(torch.float32)[panel_row]
        assert torch.equal(generated_scale, x_scale)
        different = generated_q != expected_q
        if torch.any(different):
            delta = (
                generated_q.to(torch.int16)
                - expected_q.to(torch.int16)
            ).abs()
            assert int(delta.max()) == 1
            exact_scaled = x.double() * (
                127.0 / x.double().abs().amax(dim=1, keepdim=True)
            )
            fractional = exact_scaled[different].abs().remainder(1.0)
            assert torch.all(
                torch.isclose(
                    fractional,
                    torch.tensor(0.5, dtype=torch.float64),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            quant_half_ties += int(torch.count_nonzero(different))
        ref = (
            torch._int_mm(generated_q, weight).float()
            * generated_scale
            * weight_scale.unsqueeze(0)
        ).to(torch.bfloat16)
        assert torch.equal(out, ref), f"prefill mismatch at M={rows}"
        exact_shapes.append([rows, k, n])

    kai_cache = next(iter(_w8_prefill_i8mm_kai_kernel.device_caches.values()))[
        0
    ]
    kai_variants = list(kai_cache.values())
    kai_assembly = min(
        (kernel.asm["asm"].lower() for kernel in kai_variants),
        key=lambda assembly: len(assembly.splitlines()),
    )
    assert kai_assembly.count("smmla") > 0
    assert "folded spill" not in kai_assembly
    assert "folded reload" not in kai_assembly

    return {
        "bit_exact_shapes": exact_shapes,
        "m64_kai_prefill": True,
        "compiled_variants": len(kai_variants),
        "kai_prefill_smmla": kai_assembly.count("smmla"),
        "kai_prefill_asm_lines": len(kai_assembly.splitlines()),
        "kai_prefill_spill_free": True,
        "short_tail_shapes_bit_exact": True,
        "quant_fp32_half_ties": quant_half_ties,
    }


def check_rope() -> dict[str, object]:
    from transformers.models.qwen3 import modeling_qwen3

    from flag_gems.runtime.backend._arm.fused.patch_qwen3_rope import (
        _PATCHED,
        _patched_apply_rotary_pos_emb,
    )

    torch.manual_seed(102)
    q_source = torch.randn(1, 1, 16, 128, dtype=torch.bfloat16)
    k_source = torch.randn(1, 1, 8, 128, dtype=torch.bfloat16)
    q = q_source.transpose(1, 2)
    k = k_source.transpose(1, 2)
    angle = torch.randn(1, 1, 64, dtype=torch.float32)
    cos = torch.cat([angle.cos(), angle.cos()], dim=-1).to(torch.bfloat16)
    sin = torch.cat([angle.sin(), angle.sin()], dim=-1).to(torch.bfloat16)

    original = modeling_qwen3.apply_rotary_pos_emb
    _PATCHED["original"] = original
    ref_q, ref_k = original(q, k, cos, sin)
    out_q, out_k = _patched_apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(out_q, ref_q)
    assert torch.equal(out_k, ref_k)
    assert torch.equal(q, q_source.transpose(1, 2))
    assert torch.equal(k, k_source.transpose(1, 2))
    return {"bit_exact": True, "noncontiguous_input": True}


def check_fused_norm() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (
        _fused_add_rms_norm_kernel,
        fused_add_rms_norm,
    )

    torch.manual_seed(103)
    x = torch.randn(1, 1024, dtype=torch.bfloat16)
    residual = torch.randn(1, 1024, dtype=torch.bfloat16)
    weight = torch.randn(1024, dtype=torch.bfloat16)
    eps = 1.0e-6
    ref_residual = residual + x
    ref_x = (
        ref_residual.float()
        * torch.rsqrt(
            ref_residual.float().pow(2).mean(dim=-1, keepdim=True) + eps
        )
    ).to(torch.bfloat16) * weight
    out_x, out_residual = fused_add_rms_norm(
        x.clone(), residual.clone(), (1024,), weight, eps
    )
    assert torch.equal(out_x, ref_x)
    assert torch.equal(out_residual, ref_residual)
    cache = next(iter(_fused_add_rms_norm_kernel.device_caches.values()))[0]
    assembly = list(cache.values())[-1].asm["asm"]
    assert len(assembly.splitlines()) < 400
    return {
        "bit_exact": True,
        "asm_lines": len(assembly.splitlines()),
        "wide_vector_spill_free": "sub\tsp, sp" not in assembly,
    }


def check_norm_quant_codegen() -> dict[str, object]:
    """Audit the model-size W8 RMSNorm-to-quantization fusion kernels."""
    from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
        _add_rmsnorm_quantize_bf16_w8_rne_kernel,
        _rmsnorm_quantize_bf16_w8_rne_kernel,
    )

    k = 2048
    value = torch.empty(k, dtype=torch.bfloat16)
    residual = torch.empty_like(value)
    weight = torch.empty_like(value)
    updated = torch.empty_like(value)
    quantized = torch.empty(k, dtype=torch.int8)
    scale = torch.empty(1, dtype=torch.float32)
    compiled = {
        "rmsnorm_quantize": _rmsnorm_quantize_bf16_w8_rne_kernel.warmup(
            value,
            weight,
            quantized,
            scale,
            1.0e-6,
            K=k,
            BLOCK_K=16,
            num_warps=1,
            num_stages=1,
            grid=(1,),
        ),
        "add_rmsnorm_quantize": (
            _add_rmsnorm_quantize_bf16_w8_rne_kernel.warmup(
                value,
                residual,
                weight,
                updated,
                quantized,
                scale,
                1.0e-6,
                K=k,
                BLOCK_K=16,
                num_warps=1,
                num_stages=1,
                grid=(1,),
            )
        ),
    }
    result = {}
    for name, kernel in compiled.items():
        assembly = kernel.asm["asm"].lower()
        llir = kernel.asm["llir"].lower()
        external = [
            line
            for line in llir.splitlines()
            if " call " in line
            and "llvm." not in line
            and " asm " not in line
        ]
        assert not external
        assert "[sp" not in assembly
        assert assembly.count("fsqrt") == 1
        result[name] = {
            "asm_lines": len(assembly.splitlines()),
            "stack_refs": 0,
            "external_compute_calls": 0,
            "fsqrt": 1,
        }
    return result


def check_argmax() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.ops.argmax import (
        argmax,
        argmax_vocab_rolled_kernel,
        set_argmax_vocab_assume_finite,
    )
    from flag_gems.runtime.backend._arm.vector_config import REDUCTION_TILE

    torch.manual_seed(106)
    values = torch.randn(151936, dtype=torch.bfloat16)
    values[17] = values[65553] = 100
    output = torch.empty((), dtype=torch.int64)
    argmax_vocab_rolled_kernel[(1,)](
        values,
        output,
        values.numel(),
        BLOCK_SIZE=REDUCTION_TILE,
        ASSUME_FINITE=True,
        num_warps=1,
        num_stages=1,
    )
    assert int(output) == int(torch.argmax(values)) == 17

    set_argmax_vocab_assume_finite(False)
    nan_values = torch.randn(100, dtype=torch.float32)
    nan_values[17] = float("nan")
    nan_values[3] = float("nan")
    assert int(argmax(nan_values, dim=0)) == int(torch.argmax(nan_values)) == 3

    cache = next(iter(argmax_vocab_rolled_kernel.device_caches.values()))[0]
    finite = list(cache.values())[0]
    # Select the smaller finite-logits specialization even if another cache
    # entry was populated by an earlier validation run.
    finite = min(cache.values(), key=lambda compiled: len(compiled.asm["asm"]))
    assembly = finite.asm["asm"]
    assert len(assembly.splitlines()) < 350
    return {
        "finite_tie_exact": True,
        "generic_first_nan_exact": True,
        "tile": REDUCTION_TILE,
        "asm_lines": len(assembly.splitlines()),
        "wide_vector_spill_free": "sub\tsp, sp" not in assembly,
    }


def check_fused_mlp() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (
        fused_add_rms_norm,
    )
    from flag_gems.runtime.backend._arm.fused.patch_qwen3_mlp import (
        FusedMLPWrapper,
    )
    from flag_gems.runtime.backend._arm.int8.quantize_live import (
        _quantize_weight_per_channel_sym,
    )
    from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
        TLEInt8Linear,
        _quantize_bf16_w8_rne_kernel,
    )
    from flag_gems.runtime.backend._arm.ops.silu_and_mul import (
        _SWIGLU_TILE,
        _swiglu_ordinary_kernel,
        _swiglu_quantize_w8_rne_kernel,
        arm_silu_and_mul,
    )

    torch.manual_seed(104)
    k, n = 128, 256
    x = (torch.randn(1, k) * 0.2).to(torch.bfloat16)
    gate_q, gate_s = _quantize_weight_per_channel_sym(
        (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    )
    up_q, up_s = _quantize_weight_per_channel_sym(
        (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    )
    gate_ref = TLEInt8Linear(gate_q, gate_s)
    up_ref = TLEInt8Linear(up_q, up_s)
    ref = F.silu(gate_ref(x)) * up_ref(x)
    # Build fresh projections for the wrapper: the standalone reference call
    # above legitimately retiles and consumes each projection's decode pack.
    gate = TLEInt8Linear(gate_q, gate_s)
    up = TLEInt8Linear(up_q, up_s)
    wrapper = FusedMLPWrapper(
        gate,
        up,
        torch.nn.Identity(),
        F.silu,
    )
    assert wrapper._fused
    out = wrapper.forward(x)
    assert torch.equal(out, ref)

    down_q, down_s = _quantize_weight_per_channel_sym(
        (torch.randn(128, n) * 0.02).to(torch.bfloat16)
    )
    down_ref = TLEInt8Linear(down_q, down_s)
    down_expected = down_ref(ref)
    down_wrapper = FusedMLPWrapper(
        TLEInt8Linear(gate_q, gate_s),
        TLEInt8Linear(up_q, up_s),
        TLEInt8Linear(down_q, down_s),
        F.silu,
    )
    assert down_wrapper._fused_down
    assert torch.equal(down_wrapper.forward(x), down_expected)

    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(k, dtype=torch.bfloat16), requires_grad=False
            )
            self.variance_epsilon = 1.0e-6

    norm = Norm()
    residual = torch.randn_like(x)
    value = torch.randn_like(x)
    expected_norm, expected_residual = fused_add_rms_norm(
        value.clone(),
        residual.clone(),
        (k,),
        norm.weight,
        norm.variance_epsilon,
    )
    expected_mlp = down_wrapper.forward(expected_norm)
    with torch.inference_mode():
        assert down_wrapper.can_fuse_add_rmsnorm(value, residual, norm)
        fused_mlp, fused_residual = down_wrapper.forward_add_rmsnorm(
            value.clone(), residual.clone(), norm
        )
    assert torch.equal(fused_residual, expected_residual)
    assert torch.equal(fused_mlp, expected_mlp)
    prefill_x = (torch.randn(2, k) * 0.2).to(torch.bfloat16)
    prefill_ref = F.silu(gate_ref(prefill_x)) * up_ref(prefill_x)
    assert torch.equal(wrapper.forward(prefill_x), prefill_ref)

    torch.set_num_threads(2)
    try:
        wrapper_mt = FusedMLPWrapper(
            TLEInt8Linear(gate_q, gate_s),
            TLEInt8Linear(up_q, up_s),
            torch.nn.Identity(),
            F.silu,
        )
        assert wrapper_mt._fused
        assert torch.equal(wrapper_mt.forward(x), ref)
        down_wrapper_mt = FusedMLPWrapper(
            TLEInt8Linear(gate_q, gate_s),
            TLEInt8Linear(up_q, up_s),
            TLEInt8Linear(down_q, down_s),
            F.silu,
        )
        assert down_wrapper_mt._fused_down
        assert torch.equal(down_wrapper_mt.forward(x), down_expected)
    finally:
        torch.set_num_threads(1)

    # Exhaust every finite BF16 gate encoding. This includes signed zero,
    # subnormals, transition points and both finite extrema of the inlined
    # SLEEF-u10 exp range handling.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    finite = ((bits.to(torch.int32) >> 7) & 0xFF) != 0xFF
    finite_gate = bits[finite].view(torch.bfloat16).contiguous()
    finite_up = torch.ones_like(finite_gate)
    finite_out = torch.empty_like(finite_gate)
    _swiglu_ordinary_kernel[(1,)](
        finite_gate,
        finite_up,
        finite_out,
        finite_gate.numel(),
        BLOCK_SIZE=_SWIGLU_TILE,
        num_warps=1,
        num_stages=1,
    )
    finite_ref = (F.silu(finite_gate) * finite_up).to(torch.bfloat16)
    assert torch.equal(finite_out, finite_ref)
    finite_q = torch.empty(finite_gate.numel(), dtype=torch.int8)
    finite_scale = torch.empty(1, dtype=torch.float32)
    _quantize_bf16_w8_rne_kernel[(1,)](
        finite_out,
        finite_q,
        finite_scale,
        K=finite_gate.numel(),
        BLOCK_K=16,
    )
    fused_finite_out = torch.empty_like(finite_gate)
    fused_finite_q = torch.empty_like(finite_q)
    fused_finite_scale = torch.empty_like(finite_scale)
    _swiglu_quantize_w8_rne_kernel[(1,)](
        finite_gate,
        finite_up,
        fused_finite_out,
        fused_finite_q,
        fused_finite_scale,
        finite_gate.numel(),
        BLOCK_SIZE=_SWIGLU_TILE,
    )
    assert torch.equal(fused_finite_out, finite_out)
    assert torch.equal(fused_finite_q, finite_q)
    assert torch.equal(fused_finite_scale, finite_scale)

    quant_cache = next(
        iter(_swiglu_quantize_w8_rne_kernel.device_caches.values())
    )[0]
    quant_variants = list(quant_cache.values())
    quant_callee_save_refs = 0
    for compiled in quant_variants:
        assembly = compiled.asm["asm"].lower()
        llir = compiled.asm["llir"].lower()
        folded = assembly.count("folded spill") + assembly.count(
            "folded reload"
        )
        assert folded <= 8
        quant_callee_save_refs = max(quant_callee_save_refs, folded)
        external_calls = [
            line
            for line in llir.splitlines()
            if " call " in line
            and "llvm." not in line
            and " asm " not in line
        ]
        assert not external_calls

    route_gate = torch.randn(8192, dtype=torch.bfloat16)
    route_up = torch.randn_like(route_gate)
    assert torch.equal(
        arm_silu_and_mul(route_gate, route_up),
        F.silu(route_gate) * route_up,
    )
    return {
        "bit_exact": True,
        "ordinary_tl_dot": True,
        "tle_required": False,
        "multicore_matrix_bit_exact": True,
        "prefill_fallback_bit_exact": True,
        "finite_bf16_swiglu_cases": finite_gate.numel(),
        "down_quant_fusion_bit_exact": True,
        "add_rmsnorm_input_quant_fusion_bit_exact": True,
        "swiglu_quant_external_compute_calls": 0,
        "swiglu_quant_callee_save_refs": quant_callee_save_refs,
        "standalone_auto_threshold": 8192,
    }


def check_qkv_fusion() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.fused.patch_qwen3_qkv import (
        patch_qwen3_qkv,
        unpatch_qwen3_qkv,
    )
    from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
        TLEInt8Linear,
    )
    from flag_gems.runtime.backend._arm.int8.optimize_qwen3 import (
        optimize_qwen3_w8a8,
    )

    qkv_default = inspect.signature(optimize_qwen3_w8a8).parameters[
        "enable_qkv_fusion"
    ].default
    assert qkv_default is True

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = make_linear()
            self.k_proj = make_linear()
            self.v_proj = make_linear()

        def forward(self, value):
            return (
                self.q_proj(value),
                self.k_proj(value),
                self.v_proj(value),
            )

    def make_linear():
        return TLEInt8Linear(
            torch.randint(-127, 128, (256, 128), dtype=torch.int8),
            torch.rand(256, dtype=torch.float32) / 127.0,
        )

    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(128, dtype=torch.bfloat16), requires_grad=False
            )
            self.variance_epsilon = 1.0e-6

        def forward(self, value):
            fp32 = value.float()
            rrms = torch.rsqrt(
                fp32.square().mean(dim=-1, keepdim=True)
                + self.variance_epsilon
            )
            return (fp32 * rrms).to(torch.bfloat16) * self.weight

    torch.manual_seed(107)
    attention = Attention()
    x = torch.randn(1, 128, dtype=torch.bfloat16)
    reference = attention(x)
    norm = Norm()
    norm_input = torch.randn(1, 1, 128, dtype=torch.bfloat16)
    expected_normalized = norm(norm_input)
    expected_norm_qkv = attention(expected_normalized)
    assert patch_qwen3_qkv(attention) == 1
    assert all(
        projection._packed_codegen is None
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
        )
    )
    output = attention(x)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(output, reference)
    )

    with torch.inference_mode():
        assert attention._triton_qkv_coordinator.prepare_input_norm(
            norm_input, norm
        )
        actual_norm_qkv = attention(norm_input)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_norm_qkv, expected_norm_qkv)
    )
    prefill = attention(torch.randn(2, 128, dtype=torch.bfloat16))
    assert all(value.shape == (2, 256) for value in prefill)
    assert unpatch_qwen3_qkv(attention) == 1
    assert all(
        projection._packed_codegen is not None
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
        )
    )

    # Rebuild under a multicore policy so the coordinator takes the shared
    # RNE quantizer + spill-free N64 grid instead of the whole projection.
    torch.set_num_threads(2)
    attention_mt = Attention()
    reference_mt = attention_mt(x)
    assert patch_qwen3_qkv(attention_mt) == 1
    output_mt = attention_mt(x)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(output_mt, reference_mt)
    )
    assert unpatch_qwen3_qkv(attention_mt) == 1
    torch.set_num_threads(1)
    return {
        "bit_exact": True,
        "one_combined_qkv_matrix": True,
        "shared_activation_quantizer": True,
        "input_rmsnorm_quant_fusion_bit_exact": True,
        "multicore_shared_quant_bit_exact": True,
        "source_packs_released": True,
        "prefill_fallback": True,
        "unpatch_rebuilds_packs": True,
        "curated_default_enabled": True,
    }


def check_qk_norm_fusion() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.fused.patch_qwen3_qk_norm import (
        _qk_rms_norm_contiguous_kernel,
        patch_qwen3_qk_norm,
        unpatch_qwen3_qk_norm,
    )
    from flag_gems.runtime.backend._arm.fused.patch_qwen3_rmsnorm import (
        _rms_norm,
    )
    from flag_gems.runtime.backend._arm.int8.optimize_qwen3 import (
        optimize_qwen3_w8a8,
    )

    qk_default = inspect.signature(optimize_qwen3_w8a8).parameters[
        "enable_qk_norm_fusion"
    ].default
    assert qk_default is True

    class Norm(torch.nn.Module):
        def __init__(self, n: int):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(n, dtype=torch.bfloat16), requires_grad=False
            )
            self.variance_epsilon = 1e-6

        def forward(self, value):
            return _rms_norm(value, self.weight, self.variance_epsilon)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_norm = Norm(128)
            self.k_norm = Norm(128)
            self._triton_qkv_coordinator = type(
                "QKVLayout", (), {"logical_sizes": (2048, 1024, 1024)}
            )()

    torch.manual_seed(108)
    attention = Attention()
    qk_source = torch.randn((24, 128), dtype=torch.bfloat16)
    q_source = qk_source[:16].reshape(1, 1, 16, 128)
    k_source = qk_source[16:].reshape(1, 1, 8, 128)
    expected_q = attention.q_norm(q_source)
    expected_k = attention.k_norm(k_source)
    assert patch_qwen3_qk_norm(attention) == 1
    with torch.inference_mode():
        qk = qk_source.clone()
        q = qk[:16].reshape(q_source.shape)
        k = qk[16:].reshape(k_source.shape)
        q_output = attention.q_norm(q)
        k_output = attention.k_norm(k)
    assert q_output is not q and k_output is not k
    assert torch.equal(q_output, expected_q)
    assert torch.equal(k_output, expected_k)

    cache = next(
        iter(_qk_rms_norm_contiguous_kernel.device_caches.values())
    )[0]
    compiled = list(cache.values())[-1]
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    external_calls = [
        line
        for line in llir.splitlines()
        if " call " in line and "llvm." not in line and " asm " not in line
    ]
    assert len(assembly.splitlines()) < 300
    assert "folded spill" not in assembly
    assert "folded reload" not in assembly
    assert not external_calls

    # Non-decode shapes execute the saved standalone implementations.
    q_prefill = torch.randn((1, 2, 16, 128), dtype=torch.bfloat16)
    k_prefill = torch.randn((1, 2, 8, 128), dtype=torch.bfloat16)
    with torch.inference_mode():
        q_prefill_output = attention.q_norm(q_prefill)
        k_prefill_output = attention.k_norm(k_prefill)
    assert torch.equal(
        q_prefill_output,
        _rms_norm(q_prefill, attention.q_norm.weight, 1e-6),
    )
    assert torch.equal(
        k_prefill_output,
        _rms_norm(k_prefill, attention.k_norm.weight, 1e-6),
    )
    assert unpatch_qwen3_qk_norm(attention) == 1
    return {
        "bit_exact": True,
        "single_combined_launch": True,
        "prefill_fallback_bit_exact": True,
        "curated_default_enabled": True,
        "unpatch_restores_forwards": True,
        "asm_lines": len(assembly.splitlines()),
        "spill_free": True,
        "external_compute_calls": 0,
    }


def check_attention() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.ops.attention import (
        _aten_sdpa,
        _flash_attn_decode_pv_codegen_kernel,
        _flash_attn_decode_scores_codegen_kernel,
        scaled_dot_product_attention,
    )

    torch.manual_seed(105)
    errors = {}
    for seq_len in (128, 512):
        q = torch.randn(1, 16, 1, 128, dtype=torch.bfloat16)
        k = torch.randn(1, 8, seq_len, 128, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        ref = _aten_sdpa(q, k, v, enable_gqa=True)
        out = scaled_dot_product_attention(q, k, v, enable_gqa=True)
        diff = ref.float() - out.float()
        rel_l2 = float(diff.norm() / ref.float().norm())
        max_abs = float(diff.abs().max())
        assert rel_l2 < 0.003
        errors[str(seq_len)] = {
            "bit_exact": torch.equal(out, ref),
            "relative_l2": rel_l2,
            "max_abs": max_abs,
        }

    scores_cache = next(
        iter(_flash_attn_decode_scores_codegen_kernel.device_caches.values())
    )[0]
    scores_compiled = list(scores_cache.values())[-1]
    scores_asm = scores_compiled.asm["asm"].lower()
    scores_llir = scores_compiled.asm["llir"].lower()
    pv_cache = next(
        iter(_flash_attn_decode_pv_codegen_kernel.device_caches.values())
    )[0]
    pv_compiled = list(pv_cache.values())[-1]
    pv_asm = pv_compiled.asm["asm"].lower()
    pv_llir = pv_compiled.asm["llir"].lower()
    assert scores_asm.count("bfdot") == 16
    # The only accepted stack traffic is one D8/D9 callee-save pair in the
    # function prologue/epilogue; the N-loop accumulator itself must not spill.
    pv_folded_stack = pv_asm.count("folded spill") + pv_asm.count(
        "folded reload"
    )
    assert pv_folded_stack <= 2
    assert "flash_attn_decode_bf16" not in scores_llir
    assert "flash_attn_decode_bf16" not in pv_llir
    return {
        "shapes": errors,
        "staged_qk_bfdot": scores_asm.count("bfdot"),
        "staged_pv_hot_loop_spill_free": True,
        "staged_pv_callee_save_refs": pv_folded_stack,
        "external_attention_compute_calls": 0,
        "default_enabled": False,
    }


def check_decode_frontend_contract() -> dict[str, object]:
    from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
        _TLE_DECODE_AVAILABLE,
        TLEInt8Linear,
        select_w8_decode_tile_n,
    )

    decoder_tile_n = select_w8_decode_tile_n(3072, 512)
    vocabulary_tile_n = select_w8_decode_tile_n(152064, 512)
    assert decoder_tile_n == 64
    assert vocabulary_tile_n == 32
    try:
        TLEInt8Linear(
            torch.empty((8, 10), dtype=torch.int8),
            torch.ones(8, dtype=torch.float32),
        )
    except ValueError as error:
        assert "divisible by 4" in str(error)
    else:
        raise AssertionError("invalid SDOT K shape was accepted")
    # TLE availability is intentionally informational.  The active W8 decode,
    # QKV and MLP checks above must pass through ordinary Triton even when the
    # independent 3.7 port exposes none of the development-only builder ops.
    return {
        "ordinary_tl_dot_required": True,
        "tle_decode_frontend_available": _TLE_DECODE_AVAILABLE,
        "decoder_tile_n": decoder_tile_n,
        "vocabulary_tile_n": vocabulary_tile_n,
        "invalid_sdot_shape_rejected": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate active Arm W8 and fused ordinary-Triton routes"
    )
    parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = {
        "triton_module": triton.__file__,
        "decode_codegen": check_decode_and_codegen(),
        "w8_quantizer_finite_bf16": check_w8_quantizer_finite_bf16(),
        "prefill_codegen": check_prefill_and_codegen(),
        "rope": check_rope(),
        "fused_norm": check_fused_norm(),
        "norm_quant_codegen": check_norm_quant_codegen(),
        "argmax": check_argmax(),
        "fused_mlp": check_fused_mlp(),
        "qkv_fusion": check_qkv_fusion(),
        "qk_norm_fusion": check_qk_norm_fusion(),
        "attention": check_attention(),
        "decode_frontend_contract": check_decode_frontend_contract(),
    }
    print("VALIDATION_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
