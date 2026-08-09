"""Hybrid vLLM W8 backend with measured generated/KAI shape cutoffs.

The decode path calls two shape-specialized shared objects generated from
ordinary Triton: BF16->qai8dxp packing and an N4/K8 SDOT matrix kernel with a
BF16 store. The C++ dispatcher contains no quantization or matrix arithmetic.
Generated M8/M12/M16 paths use an MR4 BF16 pack and shape-specialized I8MM
matrix kernels. M4 remains on KAI because it loses in every measured production
shape. Decode and prefill have independent thread cutoffs. Above both cutoffs
the original KAI closure is installed at model load; other batch sizes also
remain on the existing KleidiAI path.
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import importlib.util
import logging
import os
import platform
import sys
import time
from pathlib import Path

import torch

from vllm_fl.ops import cpu_int8_kai as _kai

logger = logging.getLogger("vllm_fl.cpu_int8_triton_codegen")
STATS = {
    "int8_linears": 0,
    "codegen_shapes": 0,
    "m1_codegen_calls": 0,
    "m4_codegen_calls": 0,
    "m8_codegen_calls": 0,
    "m12_codegen_calls": 0,
    "m16_codegen_calls": 0,
    "kai_fallback_calls": 0,
}
for _profile_m in (1, 4, 8, 12, 16):
    STATS[f"m{_profile_m}_codegen_launch_ns"] = 0
    STATS[f"m{_profile_m}_codegen_op_ns"] = 0
STATS["kai_fallback_launch_ns"] = 0
STATS["kai_fallback_op_ns"] = 0
PROFILE_TIME = os.environ.get("FL_CPU_INT8_TRITON_PROFILE_TIME", "0") == "1"
P0_ENABLED = os.environ.get("FL_CPU_P0_TRITON", "0") == "1"
INCLUDE_LM_HEAD = os.environ.get("FL_INT8_LMHEAD", "0") == "1"
STRICT = os.environ.get("FL_CPU_INT8_STRICT", "1") != "0"
M1_MAX_THREADS = int(
    os.environ.get("FL_CPU_INT8_TRITON_M1_MAX_THREADS", "1")
)
PREFILL_MAX_THREADS = int(
    os.environ.get(
        "FL_CPU_INT8_TRITON_PREFILL_MAX_THREADS",
        os.environ.get("FL_CPU_INT8_TRITON_M16_MAX_THREADS", "1"),
    )
)
if M1_MAX_THREADS < 0 or PREFILL_MAX_THREADS < 0:
    raise ValueError(
        "FL_CPU_INT8_TRITON_M1_MAX_THREADS and "
        "FL_CPU_INT8_TRITON_PREFILL_MAX_THREADS must be non-negative"
    )

_HERE = Path(__file__).resolve().parent
_LIBRARY = Path(
    os.environ.get(
        "FL_CPU_INT8_TRITON_LIBRARY",
        str(_HERE / "libtriton_kai_w8_decode_backend.so"),
    )
)
_BUNDLE = Path(os.environ.get("FL_CPU_INT8_TRITON_BUNDLE", ""))
if not _LIBRARY.is_file():
    raise FileNotFoundError(f"missing generated W8 dispatcher: {_LIBRARY}")
if not _BUNDLE.is_dir():
    raise FileNotFoundError(
        "FL_CPU_INT8_TRITON_BUNDLE must name the generated W8 AOT bundle"
    )

_CODEGEN = ctypes.CDLL(str(_LIBRARY))
_CODEGEN.triton_kai_w8_backend_abi_version.restype = ctypes.c_uint32
_CODEGEN.triton_kai_w8_decode_kernel_create.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int64,
    ctypes.c_int64,
]
_CODEGEN.triton_kai_w8_decode_kernel_create.restype = ctypes.c_void_p
_CODEGEN.triton_kai_w8_decode_kernel_destroy.argtypes = [ctypes.c_void_p]
_CODEGEN.triton_kai_w8_decode_launch.argtypes = [ctypes.c_void_p] * 4
_CODEGEN.triton_kai_w8_decode_launch.restype = ctypes.c_int
_CODEGEN.triton_kai_w8_prefill_launch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_CODEGEN.triton_kai_w8_prefill_launch.restype = ctypes.c_int
_CODEGEN.triton_kai_w8_decode_last_error.restype = ctypes.c_char_p
_BUNDLE_FORMAT = 2
_BACKEND_ABI = 2
_KAI_RHS_ABI = "qsi8cxp4x8-nr4-kr8-sr1"
_KAI_LHS_ABI = "qai8dxp-mr1-mr4-kr8-sr1"
# Direct same-blob measurements keep these decode shapes on KAI.  Their M8+
# prefill objects remain useful, so they stay in the bundle and are gated only
# when M==1.
_M1_KAI_SHAPES = frozenset({(1024, 1024), (2048, 1024)})  # (N, K)
_BUNDLE_FILES = (
    ("pack_sha256", "_pack_lhs_qai8dxp_bf16_kernel.so"),
    ("matrix_sha256", "_kai_w8_layout_pointer_kernel.so"),
    ("pack_m16_sha256", "_pack_lhs_qai8dxp_bf16_mr4_kernel.so"),
    ("prefill_m4_sha256", "_kai_w8_prefill_m4_kernel.so"),
    ("prefill_m8_sha256", "_kai_w8_prefill_m8_kernel.so"),
    ("prefill_m12_sha256", "_kai_w8_prefill_m12_kernel.so"),
    ("prefill_m16_sha256", "_kai_w8_prefill_kernel.so"),
)


def _read_properties(path: Path) -> dict[str, str]:
    properties: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read generated W8 metadata: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in properties:
            raise RuntimeError(
                f"invalid generated W8 metadata at {path}:{line_number}"
            )
        properties[key] = value
    return properties


def _normalize_arch(value: str) -> str:
    if value.lower() in ("aarch64", "arm64", "armv8"):
        return "aarch64"
    return value.lower()


def _runtime_target() -> tuple[str, str, set[str], int]:
    try:
        from triton.compiler.compiler import make_backend
        from triton.runtime.driver import driver

        backend = make_backend(driver.active.get_current_target())
        features = set(backend.cpu_features)
        arch = backend.cpu_arch
        sve_bits = backend.sve_vector_bits
    except (AttributeError, ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "generated W8 bundle validation requires the patched Triton-CPU backend"
        ) from exc
    return platform.system().lower(), _normalize_arch(arch), features, sve_bits


def _load_bundle_manifest() -> dict[tuple[int, int], dict[str, str]]:
    metadata = _read_properties(_BUNDLE / "bundle.meta")
    try:
        format_version = int(metadata["format_version"])
        metadata_abi = int(metadata["backend_abi"])
        expected_shapes = int(metadata["shapes"])
        target_os = metadata["target_os"]
        target_arch = _normalize_arch(metadata["target_arch"])
        target_mode = metadata["target_mode"]
        required_features = set(
            filter(None, metadata["required_features"].split(","))
        )
        expected_sve_bits = int(metadata["sve_vector_bits"])
        kai_rhs_abi = metadata["kai_rhs_abi"]
        kai_lhs_abi = metadata["kai_lhs_abi"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError("incomplete generated W8 bundle metadata") from exc
    if format_version != _BUNDLE_FORMAT:
        raise RuntimeError(
            f"unsupported generated W8 bundle format {format_version}; "
            f"expected {_BUNDLE_FORMAT}"
        )
    dispatcher_abi = int(_CODEGEN.triton_kai_w8_backend_abi_version())
    if metadata_abi != _BACKEND_ABI or dispatcher_abi != _BACKEND_ABI:
        raise RuntimeError(
            "generated W8 ABI mismatch: "
            f"bundle={metadata_abi}, dispatcher={dispatcher_abi}, "
            f"python={_BACKEND_ABI}"
        )
    if kai_rhs_abi != _KAI_RHS_ABI or kai_lhs_abi != _KAI_LHS_ABI:
        raise RuntimeError(
            "generated W8 KleidiAI packing ABI mismatch: "
            f"rhs={kai_rhs_abi}, lhs={kai_lhs_abi}"
        )
    (
        runtime_os,
        runtime_arch,
        runtime_features,
        runtime_sve_bits,
    ) = _runtime_target()
    if target_os != runtime_os or target_arch != runtime_arch:
        raise RuntimeError(
            "generated W8 target mismatch: "
            f"bundle={target_os}/{target_arch}, "
            f"runtime={runtime_os}/{runtime_arch}"
        )
    missing = required_features - runtime_features
    if missing:
        raise RuntimeError(
            "generated W8 bundle requires unavailable CPU features: "
            + ",".join(sorted(missing))
        )
    if target_mode not in ("fixed_i8mm", "sve2_i8mm"):
        raise RuntimeError(f"invalid generated W8 target mode: {target_mode}")
    if target_mode == "sve2_i8mm" and runtime_sve_bits != expected_sve_bits:
        raise RuntimeError(
            "generated W8 SVE vector-length mismatch: "
            f"bundle={expected_sve_bits}, runtime={runtime_sve_bits}"
        )

    manifest_path = _BUNDLE / "manifest.tsv"
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read generated W8 manifest: {manifest_path}"
        ) from exc
    expected_header = [
        "#k",
        "n",
        "shape",
        *[column for column, _ in _BUNDLE_FILES],
    ]
    if not lines or lines[0].split("\t") != expected_header:
        raise RuntimeError("invalid generated W8 manifest header")
    entries: dict[tuple[int, int], dict[str, str]] = {}
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != len(expected_header):
            raise RuntimeError(
                f"invalid generated W8 manifest row {line_number}"
            )
        row = dict(zip(expected_header, fields))
        try:
            k = int(row["#k"])
            n = int(row["n"])
        except ValueError as exc:
            raise RuntimeError(
                f"invalid generated W8 shape at manifest row {line_number}"
            ) from exc
        canonical_shape = f"k{k}-n{n}"
        key = (n, k)
        if (
            k <= 0
            or n <= 0
            or row["shape"] != canonical_shape
            or key in entries
        ):
            raise RuntimeError(
                f"invalid generated W8 shape at manifest row {line_number}"
            )
        for column, _ in _BUNDLE_FILES:
            digest = row[column]
            if len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest
            ):
                raise RuntimeError(
                    f"invalid generated W8 digest at manifest row {line_number}"
                )
        entries[key] = row
    if len(entries) != expected_shapes:
        raise RuntimeError(
            "generated W8 shape count mismatch: "
            f"metadata={expected_shapes}, manifest={len(entries)}"
        )
    return entries


def _verify_shape(n: int, k: int) -> bool:
    key = (n, k)
    entry = _MANIFEST.get(key)
    if entry is None:
        return False
    if key in _VERIFIED_SHAPES:
        return True
    shape_dir = _BUNDLE / entry["shape"]
    for column, file_name in _BUNDLE_FILES:
        object_path = shape_dir / file_name
        if object_path.is_symlink() or not object_path.is_file():
            raise RuntimeError(f"missing generated W8 object: {object_path}")
        try:
            actual = hashlib.sha256(object_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(
                f"cannot read generated W8 object: {object_path}"
            ) from exc
        if actual != entry[column]:
            raise RuntimeError(
                f"generated W8 object digest mismatch: {object_path}"
            )
    _VERIFIED_SHAPES.add(key)
    return True


_MANIFEST = _load_bundle_manifest()
_VERIFIED_SHAPES: set[tuple[int, int]] = set()
_HANDLES: dict[tuple[int, int], int] = {}
_ROUTE_THREADS: int | None = None
_P0_MODULE = None


def _enable_p0_codegen() -> None:
    global _P0_MODULE
    if not P0_ENABLED or _P0_MODULE is not None:
        return
    name = "vllm_fl.ops.cpu_p0_triton_codegen"
    source = Path(__file__).with_name("cpu_p0_triton_codegen.py")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.enable_p0_codegen()
    _P0_MODULE = module


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _last_error() -> str:
    value = _CODEGEN.triton_kai_w8_decode_last_error()
    return value.decode() if value else "unknown generated W8 error"


def _create_handle(n: int, k: int) -> int | None:
    key = (n, k)
    if key in _HANDLES:
        return _HANDLES[key]
    if not _verify_shape(n, k):
        return None
    handle = _CODEGEN.triton_kai_w8_decode_kernel_create(
        str(_BUNDLE).encode(), k, n
    )
    if not handle:
        raise RuntimeError(_last_error())
    _HANDLES[key] = handle
    STATS["codegen_shapes"] += 1
    return handle


def _m1_codegen_allowed(n: int, k: int) -> bool:
    return (n, k) not in _M1_KAI_SHAPES


@atexit.register
def _destroy_handles() -> None:
    for handle in _HANDLES.values():
        _CODEGEN.triton_kai_w8_decode_kernel_destroy(handle)
    _HANDLES.clear()


@torch.library.custom_op("fl_cpu::linear_w8a8_triton_codegen", mutates_args=())
def linear_w8a8_triton_codegen(
    x: torch.Tensor, packed: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    op_start_ns = time.perf_counter_ns() if PROFILE_TIME else 0
    x_bf16 = x.to(torch.bfloat16).reshape(-1, k).contiguous()
    m = x_bf16.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16)
    handle = _HANDLES.get((n, k))
    route_threads = (
        _ROUTE_THREADS
        if _ROUTE_THREADS is not None
        else torch.get_num_threads()
    )
    if (
        m == 1
        and handle is not None
        and route_threads <= M1_MAX_THREADS
        and _m1_codegen_allowed(n, k)
    ):
        launch_start_ns = time.perf_counter_ns() if PROFILE_TIME else 0
        status = _CODEGEN.triton_kai_w8_decode_launch(
            handle, _ptr(x_bf16), _ptr(packed), _ptr(out)
        )
        launch_end_ns = time.perf_counter_ns() if PROFILE_TIME else 0
        if status:
            raise RuntimeError(_last_error())
        STATS["m1_codegen_calls"] += 1
        profile_route = "m1_codegen"
    elif (
        m in (8, 12, 16)
        and handle is not None
        and route_threads <= PREFILL_MAX_THREADS
    ):
        launch_start_ns = time.perf_counter_ns() if PROFILE_TIME else 0
        status = _CODEGEN.triton_kai_w8_prefill_launch(
            handle, m, _ptr(x_bf16), _ptr(packed), _ptr(out)
        )
        launch_end_ns = time.perf_counter_ns() if PROFILE_TIME else 0
        if status:
            raise RuntimeError(_last_error())
        STATS[f"m{m}_codegen_calls"] += 1
        profile_route = f"m{m}_codegen"
    else:
        launch_start_ns = time.perf_counter_ns() if PROFILE_TIME else 0
        _kai._LIB.fl_w8a8_linear(  # pylint: disable=protected-access
            m, n, k, _ptr(x_bf16), _ptr(packed), _ptr(out)
        )
        launch_end_ns = time.perf_counter_ns() if PROFILE_TIME else 0
        STATS["kai_fallback_calls"] += 1
        profile_route = "kai_fallback"
    result = out.reshape(*x.shape[:-1], n)
    if PROFILE_TIME:
        STATS[f"{profile_route}_launch_ns"] += (
            launch_end_ns - launch_start_ns
        )
        STATS[f"{profile_route}_op_ns"] += (
            time.perf_counter_ns() - op_start_ns
        )
    return result


@linear_w8a8_triton_codegen.register_fake
def _(x: torch.Tensor, packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n), dtype=torch.bfloat16)


def _make_cpu_linear(
    packed: torch.Tensor, n: int, k: int, has_codegen: bool
):
    route_threads = (
        _ROUTE_THREADS
        if _ROUTE_THREADS is not None
        else torch.get_num_threads()
    )
    if route_threads > max(M1_MAX_THREADS, PREFILL_MAX_THREADS):
        # Install the original closure itself, rather than entering a
        # codegen custom op that immediately falls back on every layer.  This
        # keeps the measured high-thread safety route free of an extra
        # Python/custom-op branch in the token hot path.
        return _kai._make_cpu_linear(  # pylint: disable=protected-access
            packed, n, k
        )

    def cpu_linear(x, weight, bias):
        # In a DYNAMO_TRACE_ONCE graph, branching on the warm-up value of M
        # here can permanently select the wrong prefill/decode route.  Keep
        # compiled graphs opaque and let the custom-op runtime body inspect M.
        if has_codegen and torch.compiler.is_compiling():
            out = torch.ops.fl_cpu.linear_w8a8_triton_codegen(
                x, packed, n, k
            )
        else:
            m = x.numel() // k
            use_codegen = has_codegen and (
                (
                    m == 1
                    and route_threads <= M1_MAX_THREADS
                    and _m1_codegen_allowed(n, k)
                )
                or (m in (8, 12, 16) and route_threads <= PREFILL_MAX_THREADS)
            )
            if use_codegen:
                out = torch.ops.fl_cpu.linear_w8a8_triton_codegen(
                    x, packed, n, k
                )
            else:
                STATS["kai_fallback_calls"] += 1
                out = torch.ops.fl_cpu.linear_w8a8(x, packed, n, k)
        return out + bias.to(out.dtype) if bias is not None else out

    return cpu_linear


def enable_int8(verbose: bool = True) -> None:
    global _ROUTE_THREADS
    import vllm.model_executor.layers.utils as layer_utils
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    if getattr(layer_utils, "_fl_int8_triton_codegen_enabled", False):
        return
    _enable_p0_codegen()
    _ROUTE_THREADS = torch.get_num_threads()
    original_dispatch = layer_utils.dispatch_cpu_unquantized_gemm

    def dispatch(layer, remove_weight):
        weight = getattr(layer, "weight", None)
        prefix = getattr(layer, "prefix", "") or ""
        is_lm_head = isinstance(layer, ParallelLMHead)
        is_input_embedding = type(layer) is VocabParallelEmbedding
        if (
            weight is not None
            and weight.ndim == 2
            and weight.shape[1] % 8 == 0  # KleidiAI kr alignment
            and not is_input_embedding
            and (INCLUDE_LM_HEAD or not is_lm_head)
        ):
            try:
                n, k = weight.shape
                packed = _kai._quantize_pack(weight)  # pylint: disable=protected-access
                handle = (
                    _create_handle(n, k)
                    if k % 32 == 0
                    and n % 4 == 0
                    and _ROUTE_THREADS
                    <= max(M1_MAX_THREADS, PREFILL_MAX_THREADS)
                    else None
                )
                layer.cpu_linear = _make_cpu_linear(
                    packed, n, k, handle is not None
                )
                if remove_weight:
                    layer.weight = torch.nn.Parameter(
                        torch.empty(0), requires_grad=False
                    )
                STATS["int8_linears"] += 1
                return
            except Exception as exc:
                message = (
                    f"failed to prepare generated ARM W8 weight {prefix} "
                    f"{tuple(weight.shape)}"
                )
                if STRICT:
                    raise RuntimeError(message) from exc
                logger.warning("%s; falling back to BF16: %s", message, exc)
        return original_dispatch(layer, remove_weight)

    layer_utils.dispatch_cpu_unquantized_gemm = dispatch
    layer_utils._fl_int8_triton_codegen_enabled = True
    if verbose:
        logger.info(
            "[vllm_fl] ARM W8 enabled (shape-gated generated M1 and "
            "generated M8/M12/M16 "
            "through %d/%d threads respectively; direct KleidiAI closure above both "
            "cutoffs)",
            M1_MAX_THREADS,
            PREFILL_MAX_THREADS,
        )


def stats() -> dict[str, int]:
    result = dict(STATS)
    if _P0_MODULE is not None:
        result.update(_P0_MODULE.stats())
    return result
