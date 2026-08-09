#!/usr/bin/env python3
"""Correctness and dynamic-compile smoke test for the C++ W8 router."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    linear_w8_vllm_dynamic,
    pack_weights_i8mm_kai,
    pack_weights_sdot,
    pack_weights_sdot_blocked,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2560)
    args = parser.parse_args()
    if args.n % 64 or args.k % 32:
        raise ValueError("W8 test requires N%64=0 and K%32=0")
    torch.ops.load_library(str(args.library.resolve()))
    torch.manual_seed(43)
    weight = torch.randint(-127, 128, (args.n, args.k), dtype=torch.int8)
    scale = (
        torch.rand((args.n,), dtype=torch.float32) * 0.02 + 0.0005
    ).contiguous()
    decode_rhs = pack_weights_sdot_blocked(
        pack_weights_sdot(weight.T.contiguous()), 64
    ).contiguous()
    prefill_rhs = pack_weights_i8mm_kai(weight, scale).contiguous()

    for m in (1, 2, 3, 4, 7, 8, 9, 12, 16, 20, 31):
        x = torch.randn((m, args.k), dtype=torch.bfloat16)
        expected = linear_w8_vllm_dynamic(
            x, decode_rhs, prefill_rhs, scale, args.n, args.k
        )
        actual = torch.ops.triton_jit_cpu.w8_linear(
            x, decode_rhs, prefill_rhs, scale, args.n, args.k
        )
        if not torch.equal(expected, actual):
            mismatch = int(
                (expected.view(torch.int16) != actual.view(torch.int16)).sum()
            )
            max_abs = float((expected.float() - actual.float()).abs().max())
            raise AssertionError(
                f"M={m}: {mismatch} BF16 mismatches, max_abs={max_abs}"
            )

    def routed(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.triton_jit_cpu.w8_linear(
            x, decode_rhs, prefill_rhs, scale, args.n, args.k
        )

    compiled = torch.compile(
        routed, backend="eager", fullgraph=True, dynamic=True
    )
    inputs = [
        torch.randn((m, args.k), dtype=torch.bfloat16)
        for m in (12, 1, 1)
    ]
    torch._dynamo.mark_dynamic(inputs[0], 0, min=1, max=16)
    for x in inputs:
        expected = linear_w8_vllm_dynamic(
            x, decode_rhs, prefill_rhs, scale, args.n, args.k
        )
        actual = compiled(x)
        if not torch.equal(expected, actual):
            raise AssertionError(f"compiled M={x.shape[0]} differs")
    print(
        "PASS libtriton_jit W8 C++ op: "
        f"K={args.k} N={args.n}, eager and dynamic compile BF16 bit-exact"
    )


if __name__ == "__main__":
    main()
