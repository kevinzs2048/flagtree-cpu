#!/usr/bin/env python3
"""Sweep rolled Triton CPU tile widths for decode-sized elementwise reductions.

This deliberately calls the JIT kernels directly, with all output/scratch
buffers allocated ahead of time.  The numbers therefore include the Python
Triton launcher but exclude framework allocation and routing overhead.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = Path("/home/cix/venv-fep-e2e/lib/python3.11/site-packages")
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(ROOT / "python"),
    str(VENV_SITE),
]

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (
    _fused_add_rms_norm_kernel,
)
from flag_gems.runtime.backend._arm.fused.patch_qwen3_rmsnorm import (
    _rms_norm_kernel,
)
from flag_gems.runtime.backend._arm.fused.patch_qwen3_rope import (
    _rope_qk_bf16_kernel,
)
from flag_gems.runtime.backend._arm.ops.argmax import (
    argmax_kernel_1,
    argmax_kernel_2,
)


@triton.jit
def _rms_norm_lanes_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_r,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_row = x_ptr + pid * stride_r
    out_row = out_ptr + pid * stride_r
    lanes = tl.arange(0, BLOCK_SIZE)
    square_lanes = tl.zeros((BLOCK_SIZE,), tl.float32)
    for off in range(0, n_elements, BLOCK_SIZE):
        offsets = off + lanes
        x = tl.load(
            x_row + offsets, mask=offsets < n_elements, other=0.0
        ).to(tl.float32)
        square_lanes += x * x
    rrms = 1.0 / tl.sqrt(tl.sum(square_lanes, axis=0) / n_elements + eps)
    for off in range(0, n_elements, BLOCK_SIZE):
        offsets = off + lanes
        mask = offsets < n_elements
        x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
        y = (x * rrms).to(out_ptr.dtype.element_ty) * w
        tl.store(out_row + offsets, y, mask=mask)


@triton.jit
def _argmax_rolled_kernel(
    inp,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-program CPU argmax with a native-width reduction in a rolled loop."""
    best_value = -float("inf")
    best_index = 0
    for off in range(0, n_elements, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            inp + offsets, mask=offsets < n_elements, other=-float("inf")
        )
        local_value, local_index = tl.max(
            values,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        update = local_value > best_value
        best_value = tl.where(update, local_value, best_value)
        best_index = tl.where(update, off + local_index, best_index)
    tl.store(out, best_index)


@triton.jit
def _argmax_lanes_kernel(
    inp,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Keep one running winner per SIMD lane; reduce horizontally only once."""
    lanes = tl.arange(0, BLOCK_SIZE)
    best_values = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    best_indices = tl.full((BLOCK_SIZE,), 0, tl.int64)
    for off in range(0, n_elements, BLOCK_SIZE):
        offsets = off + lanes
        values = tl.load(
            inp + offsets, mask=offsets < n_elements, other=-float("inf")
        ).to(tl.float32)
        update = values > best_values
        best_values = tl.where(update, values, best_values)
        best_indices = tl.where(update, offsets, best_indices)

    max_value = tl.max(best_values, axis=0)
    candidate = tl.where(best_values == max_value, best_indices, n_elements)
    tl.store(out, tl.min(candidate, axis=0))


def median_us(fn, *, warmup: int = 30, iterations: int = 300, batches: int = 9):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(batches):
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        samples.append((time.perf_counter_ns() - begin) / iterations / 1000.0)
    return statistics.median(samples)


def asm_stats(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    compiled = list(cache.values())[-1]
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"]
    stack_bytes = 0
    for line in assembly.splitlines():
        fields = line.strip().replace(",", "").split()
        if len(fields) >= 4 and fields[:3] == ["sub", "sp", "sp"]:
            try:
                stack_bytes = max(stack_bytes, int(fields[3].lstrip("#")))
            except ValueError:
                pass
    return {
        "asm_lines": len(assembly.splitlines()),
        "llir_lines": len(llir.splitlines()),
        "stack_bytes": stack_bytes,
        "sp_mem": sum(
            "[sp" in line and line.lstrip().startswith(("ld", "st"))
            for line in assembly.splitlines()
        ),
    }


def print_row(name, tile, latency, kernel):
    stats = asm_stats(kernel)
    print(
        f"{name:12s} tile={tile:3d} us={latency:9.3f} "
        f"asm={stats['asm_lines']:5d} llir={stats['llir_lines']:4d} "
        f"stack={stats['stack_bytes']:4d} sp_mem={stats['sp_mem']:4d}"
    )


def bench_rms(tiles):
    n = 1024
    x = torch.randn((1, n), dtype=torch.bfloat16)
    w = torch.randn(n, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    reference = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16) * w
    for tile in tiles:
        run = lambda: _rms_norm_kernel[(1,)](
            x, w, out, n, n, 1e-6, BLOCK_SIZE=tile, num_warps=1, num_stages=1
        )
        run()
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        print_row("rms_norm", tile, median_us(run), _rms_norm_kernel)
    baseline = out.clone()
    for tile in tiles:
        run = lambda: _rms_norm_lanes_kernel[(1,)](
            x,
            w,
            out,
            n,
            n,
            1e-6,
            BLOCK_SIZE=tile,
            num_warps=1,
            num_stages=1,
        )
        run()
        mismatch = int((out != baseline).sum())
        print_row("rms_lanes", tile, median_us(run), _rms_norm_lanes_kernel)
        print(f"{'':12s} tile={tile:3d} baseline_mismatch={mismatch}")


def bench_fused_rms(tiles):
    n = 1024
    source = torch.randn((1, n), dtype=torch.bfloat16)
    residual0 = torch.randn((1, n), dtype=torch.bfloat16)
    w = torch.randn(n, dtype=torch.bfloat16)
    out = torch.empty_like(source)
    residual = torch.empty_like(residual0)
    residual_ref = source + residual0
    output_ref = (
        residual_ref.float()
        * torch.rsqrt(residual_ref.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16) * w
    for tile in tiles:
        def run():
            out.copy_(source)
            residual.copy_(residual0)
            _fused_add_rms_norm_kernel[(1,)](
                out,
                residual,
                w,
                n,
                n,
                n,
                1e-6,
                BLOCK_SIZE=tile,
                num_warps=1,
                num_stages=1,
            )

        run()
        torch.testing.assert_close(residual, residual_ref, rtol=0, atol=0)
        torch.testing.assert_close(out, output_ref, rtol=1e-2, atol=3.2e-2)
        print_row(
            "fused_rms", tile, median_us(run), _fused_add_rms_norm_kernel
        )


def bench_rope(tiles):
    head_dim = 128
    q_heads = 16
    kv_heads = 8
    q0 = torch.randn((q_heads, head_dim), dtype=torch.bfloat16)
    k0 = torch.randn((kv_heads, head_dim), dtype=torch.bfloat16)
    c = torch.randn(head_dim // 2, dtype=torch.bfloat16)
    s = torch.randn(head_dim // 2, dtype=torch.bfloat16)
    q = torch.empty_like(q0)
    k = torch.empty_like(k0)

    def reference(value):
        first = value[:, : head_dim // 2]
        second = value[:, head_dim // 2 :]
        first_c = (first.float() * c.float()).to(torch.bfloat16)
        second_s = (second.float() * s.float()).to(torch.bfloat16)
        first_s = (first.float() * s.float()).to(torch.bfloat16)
        second_c = (second.float() * c.float()).to(torch.bfloat16)
        return torch.cat(
            [
                first_c.float() - second_s.float(),
                first_s.float() + second_c.float(),
            ],
            dim=-1,
        ).to(torch.bfloat16)

    q_ref = reference(q0)
    k_ref = reference(k0)
    for tile in tiles:
        def run():
            q.copy_(q0)
            k.copy_(k0)
            _rope_qk_bf16_kernel[(q_heads + kv_heads,)](
                q,
                k,
                c,
                s,
                q_heads,
                head_dim,
                head_dim // 2,
                BLOCK_HALF=tile,
                num_warps=1,
                num_stages=1,
            )

        run()
        torch.testing.assert_close(q, q_ref, rtol=1.0, atol=3.2e-2)
        torch.testing.assert_close(k, k_ref, rtol=1.0, atol=3.2e-2)
        print_row("rope_qk", tile, median_us(run), _rope_qk_bf16_kernel)


def bench_argmax(tiles):
    n = 151936
    values = torch.randn(n, dtype=torch.bfloat16)
    # Exercise the left tie break across two separate rolled tiles.
    values[17] = values[65553] = torch.tensor(100, dtype=torch.bfloat16)
    out = torch.empty((), dtype=torch.int64)
    reference = torch.argmax(values)

    for tile in tiles:
        run = lambda: _argmax_rolled_kernel[(1,)](
            values,
            out,
            n,
            BLOCK_SIZE=tile,
            num_warps=1,
            num_stages=1,
        )
        run()
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        print_row("argmax_roll", tile, median_us(run), _argmax_rolled_kernel)

    for tile in tiles:
        run = lambda: _argmax_lanes_kernel[(1,)](
            values,
            out,
            n,
            BLOCK_SIZE=tile,
            num_warps=1,
            num_stages=1,
        )
        run()
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        print_row("argmax_lanes", tile, median_us(run), _argmax_lanes_kernel)

    block_size = triton.next_power_of_2(int(n**0.5 + 0.999999))
    mid_size = triton.cdiv(n, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    mid_value = torch.empty(mid_size, dtype=values.dtype)
    mid_index = torch.empty(mid_size, dtype=torch.int64)

    def baseline():
        argmax_kernel_1[(mid_size, 1, 1)](
            values, mid_value, mid_index, n, block_size
        )
        argmax_kernel_2[(1, 1, 1)](
            mid_value, mid_index, out, mid_size, block_mid
        )

    baseline()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    print(
        f"{'argmax_2stage':12s} tile={block_size:3d} "
        f"us={median_us(baseline):9.3f}"
    )
    print(f"{'torch.argmax':12s} tile={0:3d} us={median_us(lambda: torch.argmax(values)):9.3f}")


def main():
    torch.manual_seed(37)
    tiles = (4, 8, 16, 32, 64, 128)
    bench_rms(tiles)
    bench_fused_rms(tiles)
    bench_rope((4, 8, 16, 32, 64))
    bench_argmax(tiles)


if __name__ == "__main__":
    main()
