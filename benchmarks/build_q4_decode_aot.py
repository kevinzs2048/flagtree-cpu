#!/usr/bin/env python3
"""Compile the ordinary-Triton Q4 decode pack and matrix objects for one shape."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flag_gems.runtime.backend._arm.q4.kernels import (
    _pack_lhs_qsi8d32p_decode_kernel,
    _q4_decode_sdot_kai_kernel,
)


def object_path(compiled, name: str) -> Path:
    # Triton stores the native object beside the metadata file.  Keeping this
    # lookup in the compiler driver avoids depending on a process-global cache
    # directory scan when several shapes are built concurrently.
    try:
        path = Path(compiled.metadata_group[name])
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(f"compiled object does not contain {name}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()
    if args.k <= 0 or args.n <= 0 or args.k % 32 or args.n % 4:
        raise ValueError("requires K%32=0 and N%4=0")
    unroll = 4 if args.k >= 4096 else 1
    x = torch.empty((1, args.k), dtype=torch.bfloat16)
    lhs = torch.empty(args.k // 32 * 34, dtype=torch.uint8)
    rhs = torch.empty((args.n // 4) * (args.k // 32) * 72, dtype=torch.uint8)
    output = torch.empty((1, args.n), dtype=torch.bfloat16)
    pack = _pack_lhs_qsi8d32p_decode_kernel.warmup(
        x,
        lhs.view(torch.float16),
        lhs.view(torch.int8),
        1,
        args.k,
        K=args.k,
        grid=(1,),
        num_warps=1,
        num_stages=1,
    )
    matrix = _q4_decode_sdot_kai_kernel.warmup(
        lhs,
        rhs,
        output,
        0,
        args.n // 4,
        K=args.k,
        N=args.n,
        UNROLL=unroll,
        LHS_PARTITIONED=False,
        grid=(1, 8),
        num_warps=1,
        num_stages=1,
    )
    assembly = matrix.asm["asm"].lower()
    llir = matrix.asm["llir"].lower()
    expected_sdot = 8 * unroll
    sdot_count = sum(
        line.strip().startswith("sdot") for line in assembly.splitlines()
    )
    addp_count = sum(
        line.strip().startswith("addp") for line in assembly.splitlines()
    )
    if sdot_count != expected_sdot:
        raise RuntimeError(
            f"expected {expected_sdot} SDOT, got {sdot_count}"
        )
    if addp_count != unroll:
        raise RuntimeError(
            f"expected {unroll} ADDP, got {addp_count}"
        )
    if any(" call " in line and "llvm." not in line for line in llir.splitlines()):
        raise RuntimeError("Q4 matrix object contains an external call")
    print(f"pack={object_path(pack, '_pack_lhs_qsi8d32p_decode_kernel.so')}")
    print(f"matrix={object_path(matrix, '_q4_decode_sdot_kai_kernel.so')}")
    print(f"unroll={unroll}")
    print(f"sdot={expected_sdot}")


if __name__ == "__main__":
    main()
