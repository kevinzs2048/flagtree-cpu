#!/usr/bin/env python3
"""Compare generated Triton and ACLE C function bodies through the same ABI."""

import argparse
import ctypes
import time

import torch


def time_call(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter_ns()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter_ns() - start) / iters / 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-so", required=True)
    parser.add_argument("--c-so", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=5000)
    args = parser.parse_args()

    m = n = 32
    torch.manual_seed(17)
    a = torch.randint(-128, 128, (m, args.k), dtype=torch.int8)
    b = torch.randint(-128, 128, (args.k, n), dtype=torch.int8)
    triton_out = torch.empty((m, n), dtype=torch.int32)
    c_out = torch.empty_like(triton_out)
    expected = a.to(torch.int32) @ b.to(torch.int32)

    kernel_lib = ctypes.CDLL(args.kernel_so)
    triton_fn = kernel_lib.i8mm_kernel
    triton_fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int32] * 6
    triton_fn.restype = None
    triton_args = (
        a.data_ptr(),
        b.data_ptr(),
        triton_out.data_ptr(),
        0,
        0,
        0,
        1,
        1,
        1,
    )

    c_lib = ctypes.CDLL(args.c_so)
    c_fn = c_lib.gemm_sve2_i8mm
    c_fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    c_fn.restype = None
    c_args = (a.data_ptr(), b.data_ptr(), c_out.data_ptr(), m, n, args.k)

    triton_fn(*triton_args)
    c_fn(*c_args)
    assert torch.equal(triton_out, expected)
    assert torch.equal(c_out, expected)

    triton_us = time_call(triton_fn, triton_args, args.warmup, args.iters)
    c_us = time_call(c_fn, c_args, args.warmup, args.iters)
    print(
        f"32x32x{args.k}: direct Triton={triton_us:.3f} us, "
        f"direct C={c_us:.3f} us, ratio={triton_us / c_us:.3f}x"
    )


if __name__ == "__main__":
    main()
