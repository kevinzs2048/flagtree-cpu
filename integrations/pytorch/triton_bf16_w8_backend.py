"""ctypes wrapper for the direct AOT BF16-activation W8 CPU backend."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch


def pack_w8_blocks(weight_kn: torch.Tensor, block_n: int) -> torch.Tensor:
    """Pack [K,N] as [N/BLOCK_N,K/4,BLOCK_N output,4 K]."""
    if weight_kn.dtype != torch.int8 or weight_kn.ndim != 2:
        raise TypeError("weight_kn must be a rank-2 INT8 tensor")
    k, n = weight_kn.shape
    if k % 4 or block_n % 4 or n % block_n:
        raise ValueError("packed W8 backend has invalid K/N/BLOCK_N")
    return (
        weight_kn.contiguous()
        .reshape(k // 4, 4, n // block_n, block_n)
        .permute(2, 0, 3, 1)
        .contiguous()
    )


def pack_w8_microtiles(weight_kn: torch.Tensor) -> torch.Tensor:
    """Pack the BLOCK_N=4 compatibility layout."""
    return pack_w8_blocks(weight_kn, 4)


class TritonBF16W8Backend:
    """Own one shape-specialized quant+GEMV pair and its INT8 scratch."""

    def __init__(
        self,
        library: str | Path,
        quant_kernel_dir: str | Path,
        gemv_kernel_dir: str | Path,
        k: int,
        n: int,
        block_n: int = 4,
    ) -> None:
        self.k = int(k)
        self.n = int(n)
        self.block_n = int(block_n)
        self._handle = None
        self._lib = ctypes.CDLL(str(library))
        self._lib.triton_bf16_w8_kernel_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self._lib.triton_bf16_w8_kernel_create.restype = ctypes.c_void_p
        self._lib.triton_bf16_w8_kernel_create_wide.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self._lib.triton_bf16_w8_kernel_create_wide.restype = ctypes.c_void_p
        self._lib.triton_bf16_w8_kernel_destroy.argtypes = [ctypes.c_void_p]
        self._lib.triton_bf16_w8_launch.argtypes = [ctypes.c_void_p] * 5
        self._lib.triton_bf16_w8_launch.restype = ctypes.c_int
        self._lib.triton_bf16_w8_last_error.restype = ctypes.c_char_p
        create = (
            self._lib.triton_bf16_w8_kernel_create
            if self.block_n == 4
            else self._lib.triton_bf16_w8_kernel_create_wide
        )
        create_args = [
            str(quant_kernel_dir).encode(),
            str(gemv_kernel_dir).encode(),
            self.k,
            self.n,
        ]
        if self.block_n != 4:
            create_args.append(self.block_n)
        self._handle = create(*create_args)
        if not self._handle:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        value = self._lib.triton_bf16_w8_last_error()
        return value.decode() if value else "unknown Triton BF16-W8 error"

    def close(self) -> None:
        if self._handle:
            self._lib.triton_bf16_w8_kernel_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __call__(
        self,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.numel() != self.k
        ):
            raise ValueError(f"x must be contiguous CPU BF16 with {self.k} values")
        if (
            packed_weight.device.type != "cpu"
            or packed_weight.dtype != torch.int8
            or not packed_weight.is_contiguous()
            or packed_weight.numel() != self.k * self.n
        ):
            raise ValueError("packed_weight has the wrong dtype/layout/size")
        if (
            weight_scale.device.type != "cpu"
            or weight_scale.dtype != torch.float32
            or not weight_scale.is_contiguous()
            or weight_scale.numel() < self.n
        ):
            raise ValueError("weight_scale must be contiguous CPU FP32")
        if output is None:
            output = torch.empty(self.n, dtype=torch.bfloat16)
        if (
            output.device.type != "cpu"
            or output.dtype != torch.bfloat16
            or not output.is_contiguous()
            or output.numel() != self.n
        ):
            raise ValueError("output must be contiguous CPU BF16")

        status = self._lib.triton_bf16_w8_launch(
            self._handle,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(packed_weight.data_ptr()),
            ctypes.c_void_p(weight_scale.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(self._error())
        return output


class TritonBF16W8MLPBackend:
    """Own quant, joined gate/up GEMV, and BF16 SwiGLU AOT kernels."""

    def __init__(
        self,
        library: str | Path,
        quant_kernel_dir: str | Path,
        gemv_kernel_dir: str | Path,
        activation_kernel_dir: str | Path,
        k: int,
        n: int,
        block_n: int = 64,
    ) -> None:
        self.k = int(k)
        self.n = int(n)
        self.block_n = int(block_n)
        self._handle = None
        self._lib = ctypes.CDLL(str(library))
        create = self._lib.triton_bf16_w8_mlp_kernel_create
        create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        create.restype = ctypes.c_void_p
        self._lib.triton_bf16_w8_mlp_kernel_destroy.argtypes = [
            ctypes.c_void_p
        ]
        self._lib.triton_bf16_w8_mlp_launch.argtypes = [
            ctypes.c_void_p
        ] * 5
        self._lib.triton_bf16_w8_mlp_launch.restype = ctypes.c_int
        self._lib.triton_bf16_w8_last_error.restype = ctypes.c_char_p
        self._handle = create(
            str(quant_kernel_dir).encode(),
            str(gemv_kernel_dir).encode(),
            str(activation_kernel_dir).encode(),
            self.k,
            self.n,
            self.block_n,
        )
        if not self._handle:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        value = self._lib.triton_bf16_w8_last_error()
        return value.decode() if value else "unknown Triton BF16-W8 MLP error"

    def close(self) -> None:
        if self._handle:
            self._lib.triton_bf16_w8_mlp_kernel_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __call__(
        self,
        x: torch.Tensor,
        packed_gate_up: torch.Tensor,
        gate_up_scale: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty(self.n, dtype=torch.bfloat16)
        status = self._lib.triton_bf16_w8_mlp_launch(
            self._handle,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(packed_gate_up.data_ptr()),
            ctypes.c_void_p(gate_up_scale.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(self._error())
        return output


class TritonBF16RMSBackend:
    """Own one shape-specialized ordinary-Triton BF16 RMSNorm kernel."""

    def __init__(
        self,
        library: str | Path,
        kernel_dir: str | Path,
        rows: int,
        cols: int,
    ) -> None:
        self.rows = int(rows)
        self.cols = int(cols)
        self._handle = None
        self._lib = ctypes.CDLL(str(library))
        create = self._lib.triton_bf16_rms_kernel_create
        create.argtypes = [ctypes.c_char_p, ctypes.c_int64]
        create.restype = ctypes.c_void_p
        self._lib.triton_bf16_rms_kernel_destroy.argtypes = [
            ctypes.c_void_p
        ]
        self._lib.triton_bf16_rms_launch.argtypes = [
            ctypes.c_void_p
        ] * 4
        self._lib.triton_bf16_rms_launch.restype = ctypes.c_int
        self._lib.triton_bf16_w8_last_error.restype = ctypes.c_char_p
        self._handle = create(str(kernel_dir).encode(), self.rows)
        if not self._handle:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        value = self._lib.triton_bf16_w8_last_error()
        return value.decode() if value else "unknown Triton BF16 RMSNorm error"

    def close(self) -> None:
        if self._handle:
            self._lib.triton_bf16_rms_kernel_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(x)
        status = self._lib.triton_bf16_rms_launch(
            self._handle,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(self._error())
        return output
