"""vLLM plugin for the Apple Silicon Qwen3.5 W4A8 integration."""

from __future__ import annotations

import os
import platform
from pathlib import Path

_REGISTERED = False


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _configure_darwin_safety() -> str:
    backend = (
        os.getenv(
            "TRITON_CPU_QWEN35_GDN_BACKEND",
            "torch" if platform.system() == "Darwin" else "triton",
        )
        .strip()
        .lower()
    )
    if backend not in {"torch", "triton"}:
        raise ValueError("TRITON_CPU_QWEN35_GDN_BACKEND must be 'torch' or 'triton'")
    if platform.system() == "Darwin" and backend == "triton" and not _enabled("TRITON_CPU_QWEN35_ALLOW_UNSAFE_GDN"):
        raise RuntimeError(
            "The Triton GDN path is quarantined on Darwin after a reproducible "
            "SoC watchdog reset. Set TRITON_CPU_QWEN35_GDN_BACKEND=torch, or "
            "explicitly acknowledge the risk with "
            "TRITON_CPU_QWEN35_ALLOW_UNSAFE_GDN=1."
        )
    if platform.system() == "Darwin":
        # This is read while GatedDeltaNetAttention instances are constructed.
        os.environ["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] = "0"
    return backend


def register() -> None:
    """Install the Q4 router and the safe Darwin GDN implementation once."""
    global _REGISTERED
    if _REGISTERED:
        return

    os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    os.environ.setdefault("TRITON_CPU_BACKEND", "1")
    os.environ.setdefault("TRITON_CPU_FIXED_I8MM", "1")
    if Path("/opt/homebrew/opt/libomp").is_dir():
        os.environ.setdefault("TRITON_LOCAL_LIBOMP_PATH", "/opt/homebrew/opt/libomp")

    gdn_backend = _configure_darwin_safety()
    if gdn_backend == "torch":
        from .gdn_fallback import install_vllm_gdn_fallback

        install_vllm_gdn_fallback()

    runtime = os.getenv("TRITON_CPU_QWEN35_Q4_RUNTIME", "libtriton_jit").strip()
    from flag_gems.runtime.backend._arm.q4 import enable_vllm_q4_codegen

    enable_vllm_q4_codegen(verbose=True, runtime=runtime)
    _REGISTERED = True
    print(
        f"[triton-cpu-qwen35] vLLM plugin active (q4={runtime}, gdn={gdn_backend}, packed_gdn=off)",
        flush=True,
    )


__all__ = ["register"]
