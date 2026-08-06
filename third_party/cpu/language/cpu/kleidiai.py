# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Build FlagTree's direct KleidiAI W4A8/W8A8 runtimes from vendored source.

The generated shared objects are content-addressed cache artifacts. They are
never part of the source tree or wheel. TLE kernels call the two stable public
symbols declared by the CPU dialect lowerings; model integrations use the same
libraries for one-time RHS packing.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_VENDOR = _ROOT / "_vendor" / "kleidiai"
_NATIVE = _ROOT / "_native" / "kleidiai"
_REVISION_FILE = _VENDOR / "UPSTREAM_REVISION"
_EXPORT_MAP = _NATIVE / "exports.map"

_SOURCES = {
    "w4a8": (
        "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.c",
        "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c",
        "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/"
        "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod.c",
        "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/"
        "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod_asm.S",
        "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/"
        "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm.c",
        "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/"
        "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm_asm.S",
        # Blockwise (qsi4c32) pair for grouped checkpoints.  These live in the
        # same variant as the channelwise pair on purpose: the runtime symbol is
        # loaded RTLD_GLOBAL, and a grouped body with a channelwise lm_head must
        # resolve both layouts inside one process.
        "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c",
        "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/"
        "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod.c",
        "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/"
        "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod_asm.S",
        "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/"
        "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm.c",
        "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/"
        "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm_asm.S",
    ),
    "w8a8": (
        "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c",
        "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/"
        "kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c",
        "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/"
        "kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c",
    ),
}
_NATIVE_SOURCES = {
    "w4a8": ("w4a8_runtime.c", "w4a8_pack.c"),
    "w8a8": ("w8a8_runtime.c",),
}
_CFLAGS = (
    "-O3",
    "-fPIC",
    "-fvisibility=hidden",
    "-fopenmp",
    "-march=armv8.6-a+bf16+i8mm+dotprod",
)


def _variant_sources(variant: str) -> tuple[Path, ...]:
    try:
        vendor = tuple(_VENDOR / path for path in _SOURCES[variant])
        native = tuple(_NATIVE / path for path in _NATIVE_SOURCES[variant])
    except KeyError as error:
        raise ValueError(f"unsupported KleidiAI runtime: {variant!r}") from error
    sources = native + vendor
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "FlagTree KleidiAI source set is incomplete: " + ", ".join(missing)
        )
    return sources


def upstream_revision() -> str:
    """Return the exact upstream KleidiAI Git revision carried by FlagTree."""
    return _REVISION_FILE.read_text(encoding="utf-8").strip()


def _compiler() -> str:
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        raise RuntimeError("FlagTree KleidiAI runtime requires a C compiler; set CC")
    return cc


def _source_digest(variant: str, cc: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"flagtree-kleidiai-runtime-v1\0")
    digest.update(variant.encode())
    digest.update(b"\0")
    digest.update(upstream_revision().encode())
    digest.update(b"\0")
    digest.update(platform.machine().encode())
    digest.update(b"\0")
    digest.update("\0".join(_CFLAGS).encode())
    try:
        compiler_id = subprocess.check_output(
            [cc, "--version"], text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        compiler_id = cc
    digest.update(compiler_id.encode())
    for root in (_VENDOR, _NATIVE):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(_ROOT)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def runtime_abi(variant: str) -> str:
    """Content ABI used by Triton callers to invalidate stale kernel caches."""
    return _source_digest(variant, _compiler())[:20]


def _cache_directory() -> Path:
    override = os.environ.get("TRITON_KLEIDIAI_CACHE_DIR")
    if override:
        path = Path(override).expanduser()
    else:
        cache_home = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        ).expanduser()
        path = cache_home / "triton" / "kleidiai"
    path.mkdir(parents=True, exist_ok=True)
    return path


@functools.lru_cache(maxsize=None)
def build_runtime(variant: str) -> Path:
    """Build and return a cached direct-KleidiAI runtime shared library."""
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        raise RuntimeError(
            f"FlagTree KleidiAI runtimes require AArch64; current machine is {machine}"
        )

    sources = _variant_sources(variant)
    cc = _compiler()
    digest = _source_digest(variant, cc)[:20]
    cache_dir = _cache_directory()
    output = cache_dir / f"libflagtree_kai_{variant}_{digest}.so"
    lock_path = cache_dir / f".{variant}_{digest}.lock"

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return output
        with tempfile.NamedTemporaryFile(
            prefix=f".{variant}_{digest}.", suffix=".so", dir=cache_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        command = [
            cc,
            *_CFLAGS,
            f"-I{_VENDOR}",
            "-shared",
            f"-Wl,--version-script={_EXPORT_MAP}",
            "-o",
            str(temporary_path),
            *(str(path) for path in sources),
            "-lm",
        ]
        try:
            subprocess.run(command, check=True)
            temporary_path.chmod(0o755)
            os.replace(temporary_path, output)
        finally:
            temporary_path.unlink(missing_ok=True)
    return output


__all__ = ["build_runtime", "runtime_abi", "upstream_revision"]
