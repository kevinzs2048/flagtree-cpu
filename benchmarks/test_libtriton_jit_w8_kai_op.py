#!/usr/bin/env python3
"""Correctness and dynamic-compile gate for exact qai8dxp/qsi8cxp W8."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch

from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_qsi8cxp


def quantize_qai8dxp_bf16(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = x.to(torch.bfloat16).float()
    zeros = torch.zeros((values.shape[0],), dtype=torch.float32)
    row_min = torch.minimum(values.amin(dim=1), zeros)
    row_max = torch.maximum(values.amax(dim=1), zeros)
    multiplier = torch.where(
        row_min == row_max,
        torch.ones_like(row_min),
        255.0 / (row_max - row_min),
    )
    scale = torch.where(multiplier == 0, 0, 1.0 / multiplier)
    descaled_min = row_min * multiplier
    descaled_max = row_max * multiplier
    zero_point = torch.where(
        -128.0 + descaled_min + 127.0 + descaled_max > 0,
        -128.0 - descaled_min,
        127.0 - descaled_max,
    ).clamp(-128, 127)
    zero_point = torch.round(zero_point).to(torch.int32)
    quantized = (
        torch.round(values * multiplier[:, None]).to(torch.int32)
        + zero_point[:, None]
    ).clamp(-128, 127).to(torch.int8)
    return quantized, -zero_point, scale


def reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    quantized, offset, lhs_scale = quantize_qai8dxp_bf16(x)
    dot = quantized.to(torch.int32) @ weight.to(torch.int32).T
    rhs_sum = weight.to(torch.int32).sum(dim=1)
    corrected = dot + offset[:, None] * rhs_sum[None, :]
    combined_scale = lhs_scale[:, None] * weight_scale[None, :]
    return (corrected.float() * combined_scale).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--compile-backends",
        default="eager,inductor",
        help="comma-separated torch.compile backends to gate",
    )
    args = parser.parse_args()
    if args.n % 4 or args.k % 32:
        raise ValueError("exact-KAI W8 test requires N%4=0 and K%32=0")

    torch.set_num_threads(args.threads)
    torch.ops.load_library(str(args.library.resolve()))
    torch.manual_seed(20260810)
    weight = torch.randint(-127, 128, (args.n, args.k), dtype=torch.int8)
    weight_scale = (
        torch.rand((args.n,), dtype=torch.float32) * 0.02 + 0.0005
    ).contiguous()
    rhs = pack_rhs_qsi8cxp(weight, weight_scale)

    for m in (1, 2, 3, 4, 7, 8, 9, 12, 13, 16, 20, 31, 32):
        x = torch.randn((m, args.k), dtype=torch.bfloat16)
        expected = reference(x, weight, weight_scale)
        actual = torch.ops.triton_jit_cpu.w8_linear_kai(
            x, rhs, args.n, args.k
        )
        # This PyTorch expression is a semantic oracle, not the bit-exact KAI
        # oracle: its FP32 scale arithmetic may associate differently at RNE
        # boundaries.  The C++ pipeline benchmark separately compares the
        # production pack and final BF16 output directly with KleidiAI.
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.5)

    def routed(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.triton_jit_cpu.w8_linear_kai(
            x, rhs, args.n, args.k
        )

    compile_backends = [
        item.strip() for item in args.compile_backends.split(",") if item.strip()
    ]
    if not compile_backends:
        raise ValueError("--compile-backends must contain at least one backend")
    for backend in compile_backends:
        compiled = torch.compile(
            routed, backend=backend, fullgraph=True, dynamic=True
        )
        dynamic_input = torch.randn((12, args.k), dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(dynamic_input, 0, min=1, max=32)
        for x in (
            dynamic_input,
            torch.randn((1, args.k), dtype=torch.bfloat16),
            torch.randn((20, args.k), dtype=torch.bfloat16),
        ):
            eager = routed(x)
            actual = compiled(x)
            if not torch.equal(actual, eager):
                max_abs = float((actual.float() - eager.float()).abs().max())
                raise AssertionError(
                    f"{backend} output differs from eager at M={x.shape[0]}; "
                    f"max_abs={max_abs}"
                )
            expected = reference(x, weight, weight_scale)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.5)

    print(
        "PASS exact-KAI libtriton_jit W8: "
        f"N={args.n} K={args.k}, eager M sweep and "
        f"dynamic compile ({','.join(compile_backends)}) accurate"
    )


if __name__ == "__main__":
    main()
