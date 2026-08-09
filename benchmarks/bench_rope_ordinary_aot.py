#!/usr/bin/env python3
"""Shape-specialized ordinary-Triton Q/K BF16 RoPE AOT."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from bench_bf16_w8a8_ordinary_split import last_compiled, median_us


@triton.jit
def _rope_qk_aot_kernel(
    q_ptr,
    k_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    head = tl.program_id(0)
    is_query = head < Q_HEADS
    row = tl.where(
        is_query,
        q_ptr + head * HEAD_DIM,
        k_ptr + (head - Q_HEADS) * HEAD_DIM,
    )
    half: tl.constexpr = HEAD_DIM // 2
    position = tl.load(positions_ptr).to(tl.int64)
    cache_row = cos_sin_cache_ptr + position * HEAD_DIM
    for offset in range(0, half, BLOCK_HALF):
        cols = offset + tl.arange(0, BLOCK_HALF)
        first = tl.load(row + cols).to(tl.float32)
        second = tl.load(row + half + cols).to(tl.float32)
        cosine = tl.load(cache_row + cols).to(tl.float32)
        sine = tl.load(cache_row + half + cols).to(tl.float32)
        # Match current vLLM CPU semantics: both products and the add/sub are
        # evaluated in FP32, followed by a single BF16 rounding at the store.
        rotated_first = first * cosine - second * sine
        rotated_second = first * sine + second * cosine
        tl.store(row + cols, rotated_first.to(tl.bfloat16))
        tl.store(row + half + cols, rotated_second.to(tl.bfloat16))


def rotate_reference(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    half = value.shape[-1] // 2
    first = value[:, :half]
    second = value[:, half:]
    first_out = (
        first.float() * cosine.float() - second.float() * sine.float()
    ).to(torch.bfloat16)
    second_out = (
        first.float() * sine.float() + second.float() * cosine.float()
    ).to(torch.bfloat16)
    return torch.cat((first_out, second_out), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-half", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--position", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(869)
    q = torch.randn(
        args.q_heads, args.head_dim, dtype=torch.bfloat16
    )
    k = torch.randn(
        args.kv_heads, args.head_dim, dtype=torch.bfloat16
    )
    positions = torch.tensor([args.position], dtype=torch.int64)
    cos_sin_cache = torch.randn(
        max(args.position + 1, 32), args.head_dim, dtype=torch.bfloat16
    )
    cosine = cos_sin_cache[args.position, : args.head_dim // 2]
    sine = cos_sin_cache[args.position, args.head_dim // 2 :]
    q_initial = q.clone()
    k_initial = k.clone()

    def run() -> None:
        _rope_qk_aot_kernel[(args.q_heads + args.kv_heads,)](
            q,
            k,
            positions,
            cos_sin_cache,
            Q_HEADS=args.q_heads,
            KV_HEADS=args.kv_heads,
            HEAD_DIM=args.head_dim,
            BLOCK_HALF=args.block_half,
        )

    run()
    expected_q = rotate_reference(q_initial, cosine, sine)
    expected_k = rotate_reference(k_initial, cosine, sine)
    assert torch.equal(q, expected_q)
    assert torch.equal(k, expected_k)

    latency = median_us(run, args.warmup, args.iters, args.batches)
    compiled = last_compiled(_rope_qk_aot_kernel)
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    print(
        f"PASS ordinary RoPE Hq={args.q_heads} Hkv={args.kv_heads} "
        f"D={args.head_dim}\n"
        f"python_launch_us={latency:.3f}\n"
        "bit_exact=True\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"stack_load_store={sum('[sp' in line and line.lstrip().startswith(('ld', 'st')) for line in assembly.splitlines())}\n"
        f"external_calls={sum(' call ' in line and 'llvm.' not in line for line in llir.splitlines())}"
    )


if __name__ == "__main__":
    main()
