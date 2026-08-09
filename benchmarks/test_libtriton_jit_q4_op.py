#!/usr/bin/env python3
"""Correctness and compile smoke test for the vLLM C++ Q4 router."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flag_gems.runtime.backend._arm.q4.linear import (
    linear_w4a8,
    linear_w4a8_asym_kai,
    linear_w4a8_asym_g128_kai,
    pack_rhs_qsi4c32p,
    pack_rhs_qsi4c32p_asym,
    pack_rhs_qsi4c128p_asym,
    quantize_q4_0,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--g128", action="store_true")
    parser.add_argument("--asymmetric-g32", action="store_true")
    args = parser.parse_args()
    if args.g128 and args.asymmetric_g32:
        raise ValueError("--g128 and --asymmetric-g32 are mutually exclusive")
    torch.ops.load_library(str(args.library.resolve()))
    torch.manual_seed(17)
    if args.g128:
        if args.k % 128:
            raise ValueError("--g128 requires K divisible by 128")
        quantized = torch.randint(
            -8, 8, (args.n, args.k), dtype=torch.int8
        )
        scale = (
            torch.rand((args.n, args.k // 128), dtype=torch.float32) * 0.08
            + 0.002
        ).to(torch.bfloat16)
        rhs = pack_rhs_qsi4c128p_asym(quantized, scale)
        reference = linear_w4a8_asym_g128_kai
        op = torch.ops.triton_jit_cpu.q4_linear_g128
        format_name = "G128"
    else:
        weight = torch.randn((args.n, args.k), dtype=torch.bfloat16) * 0.15
        quantized, scale = quantize_q4_0(weight)
        if args.asymmetric_g32:
            rhs = pack_rhs_qsi4c32p_asym(
                quantized, scale.to(torch.bfloat16)
            )
            reference = linear_w4a8_asym_kai
            op = torch.ops.triton_jit_cpu.q4_linear_g32_asym
            format_name = "G32 qai8dxp-A8"
        else:
            rhs = pack_rhs_qsi4c32p(quantized, scale)
            reference = linear_w4a8
            op = torch.ops.triton_jit_cpu.q4_linear
            format_name = "Q4_0"
    # M24/M28/M32 force the production G128 M12/M4/M16 prefill objects;
    # the shorter shapes cover padded M8/M16 and decode M1/M3.
    for m in (1, 3, 4, 7, 8, 12, 16, 20, 24, 28, 31, 32):
        x = torch.randn((m, args.k), dtype=torch.bfloat16)
        expected = reference(x, rhs, args.n, args.k)
        actual = op(x, rhs, args.n, args.k)
        if not torch.equal(expected, actual):
            mismatch = int((expected.view(torch.int16) != actual.view(torch.int16)).sum())
            raise AssertionError(f"M={m}: {mismatch} BF16 bit mismatches")

    def compiled(x: torch.Tensor) -> torch.Tensor:
        return op(x, rhs, args.n, args.k)

    compiled_fn = torch.compile(
        compiled, backend="eager", fullgraph=True, dynamic=True
    )
    compiled_inputs = [
        torch.randn((m, args.k), dtype=torch.bfloat16)
        for m in (12, 1, 1)
    ]
    torch._dynamo.mark_dynamic(compiled_inputs[0], 0, min=1, max=16)
    for x in compiled_inputs:
        m = x.shape[0]
        expected = reference(x, rhs, args.n, args.k)
        actual = compiled_fn(x)
        if not torch.equal(expected, actual):
            raise AssertionError(f"compiled M={m} differs")
    print(
        f"PASS libtriton_jit {format_name} Q4 C++ op: "
        f"K={args.k} N={args.n}, eager M=1..32 BF16 bit-exact, "
        "torch.compile fullgraph prefill/decode bit-exact"
    )


if __name__ == "__main__":
    main()
