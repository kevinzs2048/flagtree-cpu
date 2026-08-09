#!/usr/bin/env python3
"""Measure the fixed Python/JIT CPU-launcher cost after compilation."""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def three_pointer_probe(a_ptr, b_ptr, c_ptr):
    a = tl.load(a_ptr).to(tl.int32)
    b = tl.load(b_ptr).to(tl.int32)
    tl.store(c_ptr, a + b)


def main():
    a = torch.ones(1, dtype=torch.int8)
    b = torch.ones(1, dtype=torch.int8)
    value = torch.zeros(1, dtype=torch.int32)
    for _ in range(1000):
        three_pointer_probe[(1,)](a, b, value)
    iters = 50000
    start = time.perf_counter_ns()
    for _ in range(iters):
        three_pointer_probe[(1,)](a, b, value)
    ns = (time.perf_counter_ns() - start) / iters
    assert int(value[0]) == 2
    print(f"three-pointer Triton launch: {ns / 1000.0:.3f} us")


if __name__ == "__main__":
    main()
