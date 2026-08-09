#!/usr/bin/env python3
"""Microbenchmark fused K/V DynamicCache growth against two ATen cats.

The timed functions both allocate new contiguous key and value tensors and
copy an existing cache plus newly generated states.  The Triton path fuses the
two copies into one ordinary-Triton launch; it does not call a runtime helper.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/home/kevin/triton-opt-cpu")
VENV_SITE = Path("/home/cix/venv-fep-e2e/lib/python3.11/site-packages")
TRITON_PYTHON = Path(
    os.getenv("TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python")
)
sys.path[:0] = [str(TRITON_PYTHON), str(VENV_SITE)]

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _kv_cache_concat_codegen_kernel(
    old_k,
    new_k,
    old_v,
    new_v,
    out_k,
    out_v,
    old_seq,
    new_seq,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one batch/head K/V pair, preserving concat(dim=-2) layout.

    One program owns one complete head.  A grid program per vector tile looks
    natural on a GPU, but on Triton-CPU it creates thousands of tiny OpenMP
    tasks for a cache copy.  The rolled loops below expose vectorizable copy
    loops to LLVM while the batch/head grid supplies coarse CPU parallelism.
    """
    bh = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    old_elements = old_seq * HEAD_DIM
    new_elements = new_seq * HEAD_DIM
    total_elements = old_elements + new_elements
    old_base = bh * old_elements
    new_base = bh * new_elements
    out_base = bh * total_elements

    for start in range(0, old_elements, BLOCK):
        offsets = start + lanes
        valid = offsets < old_elements
        k = tl.load(old_k + old_base + offsets, mask=valid)
        v = tl.load(old_v + old_base + offsets, mask=valid)
        tl.store(out_k + out_base + offsets, k, mask=valid)
        tl.store(out_v + out_base + offsets, v, mask=valid)

    for start in range(0, new_elements, BLOCK):
        offsets = start + lanes
        valid = offsets < new_elements
        k = tl.load(new_k + new_base + offsets, mask=valid)
        v = tl.load(new_v + new_base + offsets, mask=valid)
        tl.store(
            out_k + out_base + old_elements + offsets, k, mask=valid
        )
        tl.store(
            out_v + out_base + old_elements + offsets, v, mask=valid
        )


def triton_concat(old_k, new_k, old_v, new_v, block: int):
    batch, heads, old_seq, head_dim = old_k.shape
    new_seq = new_k.shape[-2]
    total_seq = old_seq + new_seq
    out_shape = (batch, heads, total_seq, head_dim)
    out_k = torch.empty(out_shape, dtype=old_k.dtype)
    out_v = torch.empty(out_shape, dtype=old_v.dtype)
    grid = (batch * heads,)
    _kv_cache_concat_codegen_kernel[grid](
        old_k,
        new_k,
        old_v,
        new_v,
        out_k,
        out_v,
        old_seq,
        new_seq,
        HEAD_DIM=head_dim,
        BLOCK=block,
    )
    return out_k, out_v


def aten_concat(old_k, new_k, old_v, new_v):
    return (
        torch.cat((old_k, new_k), dim=-2),
        torch.cat((old_v, new_v), dim=-2),
    )


def median_us(fn, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        begin = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - begin) / 1e3)
        # Keep the output live through the end timestamp.
        assert result[0].numel() == result[1].numel()
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="128,512,1024,2048")
    parser.add_argument("--new-seq", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=101)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    print(
        "old_seq,aten_us,triton_us,triton_over_aten,exact,logical_mib"
    )
    for old_seq in (int(value) for value in args.seq_lens.split(",")):
        shape = (args.batch, args.heads, old_seq, args.head_dim)
        new_shape = (args.batch, args.heads, args.new_seq, args.head_dim)
        old_k = torch.randn(shape, dtype=torch.bfloat16)
        old_v = torch.randn(shape, dtype=torch.bfloat16)
        new_k = torch.randn(new_shape, dtype=torch.bfloat16)
        new_v = torch.randn(new_shape, dtype=torch.bfloat16)

        reference = aten_concat(old_k, new_k, old_v, new_v)
        actual = triton_concat(
            old_k, new_k, old_v, new_v, args.block
        )
        exact = torch.equal(reference[0], actual[0]) and torch.equal(
            reference[1], actual[1]
        )
        for _ in range(10):
            aten_concat(old_k, new_k, old_v, new_v)
            triton_concat(old_k, new_k, old_v, new_v, args.block)

        aten_us = median_us(
            lambda: aten_concat(old_k, new_k, old_v, new_v),
            args.iterations,
        )
        triton_us = median_us(
            lambda: triton_concat(
                old_k, new_k, old_v, new_v, args.block
            ),
            args.iterations,
        )
        logical_mib = (
            2
            * args.batch
            * args.heads
            * (old_seq + args.new_seq)
            * args.head_dim
            * old_k.element_size()
            / (1024 * 1024)
        )
        print(
            f"{old_seq},{aten_us:.3f},{triton_us:.3f},"
            f"{triton_us / aten_us:.3f},{str(exact).lower()},"
            f"{logical_mib:.3f}"
        )


if __name__ == "__main__":
    main()
