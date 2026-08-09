"""Ordinary-Triton AOT replacements for Qwen3 CPU decode epilogues.

Only audited BF16 M=1 Qwen3-0.6B shapes enter generated code. Other shapes
remain on the original vLLM/PyTorch path and never cross an opaque custom-op
boundary.
The shared library is a loader/launcher; all normalization and RoPE
arithmetic lives in ordinary Triton-generated objects.
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import os
import platform
import time
from pathlib import Path

import torch


PROFILE_TIME = os.environ.get("FL_CPU_INT8_TRITON_PROFILE_TIME", "0") == "1"
ENABLED_OPS = set(
    filter(
        None,
        os.environ.get(
            "FL_CPU_P0_TRITON_OPS", "rms,fused_rms,rope,swiglu"
        ).split(","),
    )
)
_VALID_OPS = {"rms", "fused_rms", "rope", "swiglu"}
if not ENABLED_OPS <= _VALID_OPS:
    raise ValueError(
        "FL_CPU_P0_TRITON_OPS contains unknown names: "
        + ",".join(sorted(ENABLED_OPS - _VALID_OPS))
    )
STATS = {
    "rms_codegen_calls": 0,
    "rms_codegen_launch_ns": 0,
    "rms_codegen_op_ns": 0,
    "fused_rms_codegen_calls": 0,
    "fused_rms_codegen_launch_ns": 0,
    "fused_rms_codegen_op_ns": 0,
    "rope_codegen_calls": 0,
    "rope_codegen_launch_ns": 0,
    "rope_codegen_op_ns": 0,
    "swiglu_codegen_calls": 0,
    "swiglu_codegen_launch_ns": 0,
    "swiglu_codegen_op_ns": 0,
    "p0_native_fallback_calls": 0,
}


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _read_properties(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read P0 bundle metadata: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise RuntimeError(
                f"invalid P0 metadata at {path}:{line_number}"
            )
        result[key] = value
    return result


def _validate_bundle(bundle: Path) -> None:
    metadata = _read_properties(bundle / "bundle.meta")
    required_keys = {
        "format_version",
        "backend_abi",
        "target_os",
        "target_arch",
        "required_features",
        "objects",
    }
    if not required_keys <= metadata.keys():
        raise RuntimeError("incomplete P0 bundle metadata")
    if metadata["format_version"] != "3" or metadata["backend_abi"] != "2":
        raise RuntimeError("unsupported P0 bundle format or launcher ABI")
    runtime_os = platform.system().lower()
    runtime_arch = platform.machine().lower()
    if runtime_arch in ("arm64", "armv8"):
        runtime_arch = "aarch64"
    target_arch = metadata["target_arch"].lower()
    if target_arch in ("arm64", "armv8"):
        target_arch = "aarch64"
    if metadata["target_os"] != runtime_os or target_arch != runtime_arch:
        raise RuntimeError(
            "P0 bundle target mismatch: "
            f"bundle={metadata['target_os']}/{target_arch}, "
            f"runtime={runtime_os}/{runtime_arch}"
        )
    try:
        from triton.compiler.compiler import make_backend
        from triton.runtime.driver import driver

        features = set(
            make_backend(driver.active.get_current_target()).cpu_features
        )
    except (AttributeError, ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "P0 target validation requires the patched Triton-CPU backend"
        ) from exc
    required_features = set(
        filter(None, metadata["required_features"].split(","))
    )
    missing = required_features - features
    if missing:
        raise RuntimeError(
            "P0 bundle requires unavailable CPU features: "
            + ",".join(sorted(missing))
        )

    expected_header = ["#kind", "rows", "cols", "shape", "symbol", "sha256"]
    try:
        lines = (bundle / "manifest.tsv").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as exc:
        raise RuntimeError("cannot read P0 bundle manifest") from exc
    if not lines or lines[0].split("\t") != expected_header:
        raise RuntimeError("invalid P0 bundle manifest header")
    expected = {
        ("rms", 1, 1024, "rms-m1-n1024-e1e-6", "_rms_norm_aot_kernel"),
        ("rms", 16, 128, "rms-m16-n128-e1e-6", "_rms_norm_aot_kernel"),
        ("rms", 8, 128, "rms-m8-n128-e1e-6", "_rms_norm_aot_kernel"),
        (
            "fused_rms",
            1,
            1024,
            "vllm-fused-rms-m1-n1024-e1e-6",
            "_vllm_fused_add_rms_aot_kernel",
        ),
        ("rope", 24, 128, "rope-hq16-hkv8-d128", "_rope_qk_aot_kernel"),
        (
            "swiglu",
            1,
            3072,
            "swiglu-n3072",
            "_bf16_swiglu_inline_exp_kernel",
        ),
    }
    seen = set()
    for line_number, line in enumerate(lines[1:], 2):
        values = line.split("\t")
        if len(values) != len(expected_header):
            raise RuntimeError(f"invalid P0 manifest row {line_number}")
        row = dict(zip(expected_header, values))
        try:
            key = (
                row["#kind"],
                int(row["rows"]),
                int(row["cols"]),
                row["shape"],
                row["symbol"],
            )
        except ValueError as exc:
            raise RuntimeError(
                f"invalid P0 shape at manifest row {line_number}"
            ) from exc
        if key not in expected or key in seen:
            raise RuntimeError(f"unexpected P0 manifest row {line_number}")
        object_path = bundle / row["shape"] / f"{row['symbol']}.so"
        if object_path.is_symlink() or not object_path.is_file():
            raise RuntimeError(
                f"missing regular P0 bundle object: {object_path}"
            )
        digest = hashlib.sha256(object_path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError(f"P0 object digest mismatch: {object_path}")
        seen.add(key)
    if seen != expected or int(metadata["objects"]) != len(expected):
        raise RuntimeError("P0 bundle object set is incomplete")


class _P0Backend:
    def __init__(self, library: Path, bundle: Path) -> None:
        if not library.is_file():
            raise FileNotFoundError(f"missing P0 dispatcher: {library}")
        if not bundle.is_dir():
            raise FileNotFoundError(f"missing P0 AOT bundle: {bundle}")
        self._lib = ctypes.CDLL(str(library))
        self._lib.triton_p0_norm_backend_abi_version.restype = ctypes.c_uint32
        if self._lib.triton_p0_norm_backend_abi_version() != 2:
            raise RuntimeError("unsupported P0 norm launcher ABI")
        _validate_bundle(bundle)
        self._lib.triton_p0_norm_kernel_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int64,
        ]
        self._lib.triton_p0_norm_kernel_create.restype = ctypes.c_void_p
        self._lib.triton_p0_norm_kernel_destroy.argtypes = [ctypes.c_void_p]
        self._lib.triton_p0_rms_launch.argtypes = [ctypes.c_void_p] * 4
        self._lib.triton_p0_rms_launch.restype = ctypes.c_int
        self._lib.triton_p0_fused_add_rms_launch.argtypes = [
            ctypes.c_void_p
        ] * 4
        self._lib.triton_p0_fused_add_rms_launch.restype = ctypes.c_int
        self._lib.triton_p0_norm_last_error.restype = ctypes.c_char_p
        self._lib.triton_p0_rope_kernel_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int64,
        ]
        self._lib.triton_p0_rope_kernel_create.restype = ctypes.c_void_p
        self._lib.triton_p0_rope_kernel_destroy.argtypes = [ctypes.c_void_p]
        self._lib.triton_p0_rope_launch.argtypes = [ctypes.c_void_p] * 5
        self._lib.triton_p0_rope_launch.restype = ctypes.c_int

        self._rms: dict[tuple[int, int], int] = {}
        if "rms" in ENABLED_OPS:
            for rows, cols in ((1, 1024), (16, 128), (8, 128)):
                shape = bundle / f"rms-m{rows}-n{cols}-e1e-6"
                self._require_object(shape / "_rms_norm_aot_kernel.so")
                handle = self._lib.triton_p0_norm_kernel_create(
                    str(shape).encode(), 0, rows
                )
                self._rms[(rows, cols)] = self._require_handle(handle)

        self._fused_rms = 0
        if "fused_rms" in ENABLED_OPS:
            fused_shape = (
                bundle
                / "vllm-fused-rms-m1-n1024-e1e-6"
            )
            self._require_object(
                fused_shape / "_vllm_fused_add_rms_aot_kernel.so"
            )
            handle = self._lib.triton_p0_norm_kernel_create(
                str(fused_shape).encode(), 1, 1
            )
            self._fused_rms = self._require_handle(handle)

        self._rope = 0
        if "rope" in ENABLED_OPS:
            rope_shape = bundle / "rope-hq16-hkv8-d128"
            self._require_object(rope_shape / "_rope_qk_aot_kernel.so")
            handle = self._lib.triton_p0_rope_kernel_create(
                str(rope_shape).encode(), 24
            )
            self._rope = self._require_handle(handle)

        self._swiglu_lib = None
        self._swiglu = None
        if "swiglu" in ENABLED_OPS:
            swiglu_path = (
                bundle
                / "swiglu-n3072/_bf16_swiglu_inline_exp_kernel.so"
            )
            self._require_object(swiglu_path)
            self._swiglu_lib = ctypes.CDLL(str(swiglu_path))
            self._swiglu = (
                self._swiglu_lib._bf16_swiglu_inline_exp_kernel
            )
            self._swiglu.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                *([ctypes.c_uint32] * 6),
            ]

    @staticmethod
    def _require_object(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"missing ordinary Triton object: {path}")

    def _error(self) -> str:
        message = self._lib.triton_p0_norm_last_error()
        return message.decode() if message else "unknown P0 AOT error"

    def _require_handle(self, handle: int | None) -> int:
        if not handle:
            raise RuntimeError(self._error())
        return handle

    def close(self) -> None:
        for handle in self._rms.values():
            self._lib.triton_p0_norm_kernel_destroy(handle)
        self._rms.clear()
        if self._fused_rms:
            self._lib.triton_p0_norm_kernel_destroy(self._fused_rms)
            self._fused_rms = 0
        if self._rope:
            self._lib.triton_p0_rope_kernel_destroy(self._rope)
            self._rope = 0

    def rms(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
        rows = x.numel() // x.shape[-1]
        status = self._lib.triton_p0_rms_launch(
            self._rms[(rows, x.shape[-1])], _ptr(x), _ptr(weight), _ptr(out)
        )
        if status:
            raise RuntimeError(self._error())

    def rope(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> None:
        if not self._rope:
            raise RuntimeError("RoPE codegen was not loaded")
        status = self._lib.triton_p0_rope_launch(
            self._rope,
            _ptr(query),
            _ptr(key),
            _ptr(positions),
            _ptr(cos_sin_cache),
        )
        if status:
            raise RuntimeError(self._error())

    def fused_rms(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> None:
        if not self._fused_rms:
            raise RuntimeError("fused RMSNorm codegen was not loaded")
        status = self._lib.triton_p0_fused_add_rms_launch(
            self._fused_rms, _ptr(x), _ptr(residual), _ptr(weight)
        )
        if status:
            raise RuntimeError(self._error())

    def swiglu(self, joined: torch.Tensor, out: torch.Tensor) -> None:
        if self._swiglu is None:
            raise RuntimeError("SwiGLU codegen was not loaded")
        self._swiglu(
            _ptr(joined), _ptr(out), 0, 0, 0, 1, 1, 1
        )

_BACKEND: _P0Backend | None = None
_PATCHED = False
_ORIGINAL_RMS_NATIVE = None
_ORIGINAL_ROPE_NATIVE = None
_ORIGINAL_ROPE_CPU = None
_ORIGINAL_SILU_NATIVE = None


@atexit.register
def _close_backend() -> None:
    global _BACKEND
    if _BACKEND is not None:
        _BACKEND.close()
        _BACKEND = None


def _record(route: str, op_start: int, launch_start: int, launch_end: int) -> None:
    STATS[f"{route}_calls"] += 1
    if PROFILE_TIME:
        STATS[f"{route}_launch_ns"] += launch_end - launch_start
        STATS[f"{route}_op_ns"] += time.perf_counter_ns() - op_start


def _rms_fallback(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    value = x.float()
    rrms = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (value * rrms).to(x.dtype) * weight


def _run_rms_codegen(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    op_start = time.perf_counter_ns() if PROFILE_TIME else 0
    out = torch.empty_like(x)
    launch_start = time.perf_counter_ns() if PROFILE_TIME else 0
    assert _BACKEND is not None
    _BACKEND.rms(x, weight, out)
    launch_end = time.perf_counter_ns() if PROFILE_TIME else 0
    _record("rms_codegen", op_start, launch_start, launch_end)
    return out


def _run_fused_rms_codegen(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> None:
    op_start = time.perf_counter_ns() if PROFILE_TIME else 0
    launch_start = time.perf_counter_ns() if PROFILE_TIME else 0
    assert _BACKEND is not None
    _BACKEND.fused_rms(x, residual, weight)
    launch_end = time.perf_counter_ns() if PROFILE_TIME else 0
    _record("fused_rms_codegen", op_start, launch_start, launch_end)


def _run_rope_codegen(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    op_start = time.perf_counter_ns() if PROFILE_TIME else 0
    launch_start = time.perf_counter_ns() if PROFILE_TIME else 0
    assert _BACKEND is not None
    _BACKEND.rope(positions, query, key, cos_sin_cache)
    launch_end = time.perf_counter_ns() if PROFILE_TIME else 0
    _record("rope_codegen", op_start, launch_start, launch_end)


def _run_swiglu_codegen(x: torch.Tensor) -> torch.Tensor:
    op_start = time.perf_counter_ns() if PROFILE_TIME else 0
    out = torch.empty((*x.shape[:-1], 3072), dtype=x.dtype)
    launch_start = time.perf_counter_ns() if PROFILE_TIME else 0
    assert _BACKEND is not None
    _BACKEND.swiglu(x, out)
    launch_end = time.perf_counter_ns() if PROFILE_TIME else 0
    _record("swiglu_codegen", op_start, launch_start, launch_end)
    return out


@torch.library.custom_op("fl_cpu::p0_rms_norm", mutates_args=())
def p0_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    eligible = (
        _BACKEND is not None
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and eps == 1.0e-6
        and (rows, x.shape[-1]) in ((1, 1024), (16, 128), (8, 128))
    )
    if not eligible:
        STATS["p0_native_fallback_calls"] += 1
        return _rms_fallback(x, weight, eps)
    return _run_rms_codegen(x, weight)


@p0_rms_norm.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "fl_cpu::p0_fused_add_rms_norm", mutates_args=("x", "residual")
)
def p0_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    eligible = (
        _BACKEND is not None
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.shape == residual.shape
        and x.numel() == 1024
        and x.shape[-1] == 1024
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and eps == 1.0e-6
    )
    if not eligible:
        updated_fp32 = x.float() + residual
        rrms = torch.rsqrt(
            updated_fp32.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        normalized = (updated_fp32 * rrms).to(x.dtype) * weight
        residual.copy_(updated_fp32.to(x.dtype))
        x.copy_(normalized)
        STATS["p0_native_fallback_calls"] += 1
        return
    _run_fused_rms_codegen(x, residual, weight)


@p0_fused_add_rms_norm.register_fake
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    return None


def _patched_rms_forward(self, x, residual=None):
    # In compiled mode these small operations fuse profitably with the
    # surrounding graph. An opaque AOT call regressed full-model throughput.
    if torch.compiler.is_compiling():
        return _ORIGINAL_RMS_NATIVE(self, x, residual)
    common_eligible = (
        _BACKEND is not None
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and self.has_weight
        and self.weight.dtype == torch.bfloat16
        and self.weight.is_contiguous()
        and self.variance_size_override is None
        and self.variance_epsilon == 1.0e-6
    )
    if common_eligible:
        rows = x.numel() // x.shape[-1]
        if (
            residual is None
            and "rms" in ENABLED_OPS
            and (rows, x.shape[-1])
            in ((1, 1024), (16, 128), (8, 128))
        ):
            return _run_rms_codegen(x, self.weight.data)
        if (
            residual is not None
            and "fused_rms" in ENABLED_OPS
            and residual.dtype == torch.bfloat16
            and residual.is_contiguous()
            and residual.shape == x.shape
            and x.numel() == 1024
            and x.shape[-1] == 1024
        ):
            _run_fused_rms_codegen(x, residual, self.weight.data)
            return x, residual
    return _ORIGINAL_RMS_NATIVE(self, x, residual)


def _run_patched_rope(self, positions, query, key, original):
    if (
        torch.compiler.is_compiling()
        or key is None
        or "rope" not in ENABLED_OPS
        or _BACKEND is None
        or positions.numel() != 1
        or positions.dtype != torch.int64
        or not positions.is_contiguous()
        or query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or not query.is_contiguous()
        or not key.is_contiguous()
        or self.head_size != 128
        or self.rotary_dim != 128
        or not self.is_neox_style
        or query.numel() != 16 * 128
        or key.numel() != 8 * 128
    ):
        return original(self, positions, query, key)
    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    if (
        cos_sin_cache.dtype != torch.bfloat16
        or not cos_sin_cache.is_contiguous()
    ):
        return original(self, positions, query, key)
    _run_rope_codegen(positions, query, key, cos_sin_cache)
    return query, key


def _patched_rope_native(self, positions, query, key=None):
    return _run_patched_rope(
        self, positions, query, key, _ORIGINAL_ROPE_NATIVE
    )


def _patched_rope_cpu(self, positions, query, key=None):
    return _run_patched_rope(
        self, positions, query, key, _ORIGINAL_ROPE_CPU
    )


def _patched_silu_forward(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _ORIGINAL_SILU_NATIVE(x)
    if (
        "swiglu" in ENABLED_OPS
        and _BACKEND is not None
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.numel() == 6144
        and x.shape[-1] == 6144
    ):
        return _run_swiglu_codegen(x)
    return _ORIGINAL_SILU_NATIVE(x)


def enable_p0_codegen() -> None:
    global _BACKEND, _PATCHED
    global _ORIGINAL_RMS_NATIVE, _ORIGINAL_ROPE_NATIVE, _ORIGINAL_ROPE_CPU
    global _ORIGINAL_SILU_NATIVE
    if _PATCHED:
        return
    bundle_value = os.environ.get("FL_CPU_P0_TRITON_BUNDLE", "")
    library_value = os.environ.get("FL_CPU_P0_TRITON_LIBRARY", "")
    if not bundle_value or not library_value:
        raise RuntimeError(
            "FL_CPU_P0_TRITON_BUNDLE and FL_CPU_P0_TRITON_LIBRARY are required"
        )
    _BACKEND = _P0Backend(
        Path(library_value).resolve(), Path(bundle_value).resolve()
    )

    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.rotary_embedding.base import (
        RotaryEmbedding,
    )

    _ORIGINAL_RMS_NATIVE = RMSNorm.forward_native
    _ORIGINAL_ROPE_NATIVE = RotaryEmbedding.forward_native
    _ORIGINAL_ROPE_CPU = RotaryEmbedding.forward_cpu
    _ORIGINAL_SILU_NATIVE = SiluAndMul.forward_native
    RMSNorm.forward_native = _patched_rms_forward
    RotaryEmbedding.forward_native = _patched_rope_native
    RotaryEmbedding.forward_cpu = _patched_rope_cpu
    SiluAndMul.forward_native = staticmethod(_patched_silu_forward)
    _PATCHED = True


def stats() -> dict[str, int]:
    return dict(STATS)
