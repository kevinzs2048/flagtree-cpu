#!/usr/bin/env python3
"""Correctness and single-thread performance probe for the SVE2 i8mm path."""

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def i8mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )
        acc += tl.dot(a, b)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
    )


def launch(a, b, c, block_m, block_n, block_k):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    i8mm_kernel[grid](
        a,
        b,
        c,
        M=m,
        N=n,
        K=k,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument(
        "--bm", type=int, default=0, help="M tile; 0 selects the full M"
    )
    parser.add_argument("--bn", type=int, default=8)
    parser.add_argument(
        "--bk", type=int, default=0, help="K tile; 0 selects the full K"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.bm == 0:
        args.bm = args.m
    if args.bk == 0:
        args.bk = args.k

    torch.manual_seed(17)
    a = torch.randint(-128, 128, (args.m, args.k), dtype=torch.int8)
    b = torch.randint(-128, 128, (args.k, args.n), dtype=torch.int8)
    c = torch.empty((args.m, args.n), dtype=torch.int32)

    launch(a, b, c, args.bm, args.bn, args.bk)
    expected = a.to(torch.int32) @ b.to(torch.int32)
    if not torch.equal(c, expected):
        delta = (c.to(torch.int64) - expected.to(torch.int64)).abs()
        mismatch = int(torch.count_nonzero(delta))
        raise AssertionError(
            f"incorrect output: mismatches={mismatch}, max_abs={int(delta.max())}"
        )

    for _ in range(args.warmup):
        launch(a, b, c, args.bm, args.bn, args.bk)
    start = time.perf_counter_ns()
    for _ in range(args.iters):
        launch(a, b, c, args.bm, args.bn, args.bk)
    elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6 / args.iters
    gops = 2.0 * args.m * args.n * args.k / elapsed_ms / 1.0e6
    print(
        f"PASS M={args.m} N={args.n} K={args.k} "
        f"BM={args.bm} BN={args.bn} BK={args.bk} "
        f"{elapsed_ms:.6f} ms {gops:.2f} GOPS"
    )


if __name__ == "__main__":
    main()
