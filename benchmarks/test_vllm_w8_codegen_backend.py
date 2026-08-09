#!/usr/bin/env python3
"""Validate the vLLM exact-KAI W8 decode dispatcher independently."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import torch


def pack_rhs(
    weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    n, k = weight.shape
    stride = 4 * k + 48
    packed = torch.zeros((n // 4, stride), dtype=torch.uint8)
    values = (
        weight.reshape(n // 4, 4, k // 8, 8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(n // 4, 4 * k)
    )
    packed[:, : 4 * k].view(torch.int8).copy_(values)
    packed[:, 4 * k : 4 * k + 16].view(torch.int32).copy_(
        weight.to(torch.int32).sum(dim=1).reshape(n // 4, 4)
    )
    packed[:, 4 * k + 16 : 4 * k + 32].view(torch.float32).copy_(
        scale.reshape(n // 4, 4)
    )
    packed[:, 4 * k + 32 : 4 * k + 48].view(torch.float32).copy_(
        bias.reshape(n // 4, 4)
    )
    return packed.reshape(-1)


def quantize_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = x.reshape(-1, x.shape[-1])
    quantized = []
    offsets = []
    reciprocals = []
    for row in rows:
        values = row.float()
        row_min = min(float(values.min()), 0.0)
        row_max = max(float(values.max()), 0.0)
        quant_scale = (
            1.0 if row_min == row_max else 255.0 / (row_max - row_min)
        )
        reciprocal = 0.0 if quant_scale == 0.0 else 1.0 / quant_scale
        descaled_min = row_min * quant_scale
        descaled_max = row_max * quant_scale
        min_error = -128.0 + descaled_min
        max_error = 127.0 + descaled_max
        zero = (
            -128.0 - descaled_min
            if min_error + max_error > 0.0
            else 127.0 - descaled_max
        )
        zero_i32 = int(round(max(-128.0, min(127.0, zero))))
        quantized.append(
            (
                torch.round(values * quant_scale).to(torch.int32)
                + zero_i32
            ).clamp(-128, 127).to(torch.int8)
        )
        offsets.append(-zero_i32)
        reciprocals.append(reciprocal)
    return (
        torch.stack(quantized),
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(reciprocals, dtype=torch.float32),
    )


def bf16_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(
            "artifacts/vllm-triton-backend/"
            "libtriton_kai_w8_decode_backend.so"
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=(
            Path(os.environ["FL_CPU_INT8_TRITON_BUNDLE"])
            if "FL_CPU_INT8_TRITON_BUNDLE" in os.environ
            else None
        ),
        help="exact target/codegen-key directory printed by the AOT builder",
    )
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--m", type=int, choices=(1, 4, 8, 12, 16), default=1)
    args = parser.parse_args()
    if args.bundle is None:
        raise ValueError("--bundle or FL_CPU_INT8_TRITON_BUNDLE is required")

    library = ctypes.CDLL(str(args.library.resolve()))
    library.triton_kai_w8_backend_abi_version.restype = ctypes.c_uint32
    if library.triton_kai_w8_backend_abi_version() != 2:
        raise RuntimeError("generated W8 dispatcher ABI mismatch")
    library.triton_kai_w8_decode_kernel_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    library.triton_kai_w8_decode_kernel_create.restype = ctypes.c_void_p
    library.triton_kai_w8_decode_kernel_destroy.argtypes = [ctypes.c_void_p]
    library.triton_kai_w8_decode_launch.argtypes = [ctypes.c_void_p] * 4
    library.triton_kai_w8_decode_launch.restype = ctypes.c_int
    library.triton_kai_w8_prefill_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.triton_kai_w8_prefill_launch.restype = ctypes.c_int
    library.triton_kai_w8_decode_last_error.restype = ctypes.c_char_p

    handle = library.triton_kai_w8_decode_kernel_create(
        str(args.bundle.resolve()).encode(), args.k, args.n
    )
    if not handle:
        raise RuntimeError(library.triton_kai_w8_decode_last_error().decode())
    try:
        torch.manual_seed(9851)
        x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
        weight = torch.randint(-127, 128, (args.n, args.k), dtype=torch.int8)
        weight_scale = torch.rand(args.n, dtype=torch.float32) * 0.01 + 0.001
        bias = torch.zeros(args.n, dtype=torch.float32)
        rhs = pack_rhs(weight, weight_scale, bias)
        output = torch.empty((args.m, args.n), dtype=torch.bfloat16)
        if args.m == 1:
            status = library.triton_kai_w8_decode_launch(
                handle,
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(rhs.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
            )
        else:
            status = library.triton_kai_w8_prefill_launch(
                handle,
                args.m,
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(rhs.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
            )
        if status:
            raise RuntimeError(
                library.triton_kai_w8_decode_last_error().decode()
            )

        q, lhs_offset, lhs_scale = quantize_reference(x)
        expected_i32 = (
            q.to(torch.int32) @ weight.to(torch.int32).T
            + lhs_offset[:, None]
            * weight.to(torch.int32).sum(dim=1)[None, :]
        )
        expected = (
            expected_i32.float()
            * (lhs_scale[:, None] * weight_scale[None, :])
        ).to(torch.bfloat16)
        max_ulp = int(bf16_ulp_distance(output, expected).max())
        if max_ulp > 1:
            mismatch = torch.nonzero(
                output.view(torch.uint16) != expected.view(torch.uint16)
            ).flatten()
            raise AssertionError(
                f"BF16 mismatch count={mismatch.numel()} "
                f"max_ulp={max_ulp} first={mismatch[:16].tolist()}"
            )
        print(
            f"PASS vLLM exact-KAI W8 M={args.m} K={args.k} N={args.n}\n"
            f"python_reference_max_bf16_ulp={max_ulp}\n"
            "wrapper_compute=false"
        )
    finally:
        library.triton_kai_w8_decode_kernel_destroy(handle)


if __name__ == "__main__":
    main()
