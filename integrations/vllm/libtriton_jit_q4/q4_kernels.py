"""Importable ordinary-Triton entry points used by the C++ JIT router."""

from flag_gems.runtime.backend._arm.q4.kernels import (
    _pack_lhs_qsi8d128p_asym_panel4_kernel,
    _pack_lhs_qai8dxp_asym_panel4_kernel,
    _pack_lhs_qsi8d32p_panel4_scalar_kernel,
    _q4_fused_decode_asym_g128_sdot_kernel,
    _q4_fused_decode_asym_g128_kai_sdot_kernel,
    _q4_fused_decode_asym_g32_kai_sdot_kernel,
    _q4_fused_decode_asym_sdot_kai_kernel,
    _q4_fused_decode_sdot_kai_kernel,
    _q4_prefill_asym_g128_i8mm_kernel,
    _q4_prefill_asym_g128_i8mm_kai_m12_k32_kernel,
    _q4_prefill_asym_i8mm_kai_kernel,
    _q4_prefill_i8mm_kai_kernel,
)

__all__ = [
    "_pack_lhs_qsi8d128p_asym_panel4_kernel",
    "_pack_lhs_qai8dxp_asym_panel4_kernel",
    "_pack_lhs_qsi8d32p_panel4_scalar_kernel",
    "_q4_fused_decode_asym_g128_sdot_kernel",
    "_q4_fused_decode_asym_g128_kai_sdot_kernel",
    "_q4_fused_decode_asym_g32_kai_sdot_kernel",
    "_q4_fused_decode_asym_sdot_kai_kernel",
    "_q4_fused_decode_sdot_kai_kernel",
    "_q4_prefill_asym_g128_i8mm_kernel",
    "_q4_prefill_asym_g128_i8mm_kai_m12_k32_kernel",
    "_q4_prefill_asym_i8mm_kai_kernel",
    "_q4_prefill_i8mm_kai_kernel",
]
