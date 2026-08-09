#!/usr/bin/env python3
"""Ordinary Triton over KAI's qsi8d32p/qsi4c32p W4 decode layout."""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _kai_w4_layout_split_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    clamp_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    groups: tl.constexpr = K // 32
    lhs_group_stride: tl.constexpr = 34
    rhs_group_stride: tl.constexpr = 72
    rhs_tile_stride: tl.constexpr = groups * rhs_group_stride
    q_lanes = tl.arange(0, 16)
    x_lanes = tl.arange(0, 8)
    output_lanes = tl.arange(0, 4)
    clamp_min = tl.load(clamp_ptr)
    clamp_max = tl.load(clamp_ptr + 1)

    for tile in range(range_begin, range_end):
        result = tl.zeros((4,), dtype=tl.float32)
        rhs_tile = tile * rhs_tile_stride
        lhs_group_ptr = lhs_packed_ptr
        rhs_group_ptr = rhs_packed_ptr + rhs_tile
        for group in tl.range(0, groups, loop_unroll_factor=UNROLL):
            lhs_scale = tl.load(
                lhs_group_ptr.to(tl.pointer_type(tl.float16))
            ).to(tl.float32)
            rhs_scale_base = rhs_group_ptr.to(
                tl.pointer_type(tl.float16)
            )
            rhs_scale = tl.load(rhs_scale_base + output_lanes).to(tl.float32)

            q0 = tl.load(rhs_group_ptr + 8 + q_lanes)
            q1 = tl.load(rhs_group_ptr + 24 + q_lanes)
            q2 = tl.load(rhs_group_ptr + 40 + q_lanes)
            q3 = tl.load(rhs_group_ptr + 56 + q_lanes)
            q0_low = (q0 << 4).to(tl.int8).reshape((4, 4))
            q1_low = (q1 << 4).to(tl.int8).reshape((4, 4))
            q2_low = (q2 << 4).to(tl.int8).reshape((4, 4))
            q3_low = (q3 << 4).to(tl.int8).reshape((4, 4))
            q0_high = (q0 & 0xF0).to(tl.int8).reshape((4, 4))
            q1_high = (q1 & 0xF0).to(tl.int8).reshape((4, 4))
            q2_high = (q2 & 0xF0).to(tl.int8).reshape((4, 4))
            q3_high = (q3 & 0xF0).to(tl.int8).reshape((4, 4))

            x_ptr = lhs_group_ptr + 2
            # KAI stores signed activation bytes in an otherwise byte-addressed
            # blob.  Make the signed interpretation explicit before widening;
            # pointer inference alone would otherwise select zero extension.
            x0 = tl.load(x_ptr + x_lanes).to(tl.int8).reshape((2, 4))
            x1 = tl.load(x_ptr + 8 + x_lanes).to(tl.int8).reshape((2, 4))
            x2 = tl.load(x_ptr + 16 + x_lanes).to(tl.int8).reshape((2, 4))
            x3 = tl.load(x_ptr + 24 + x_lanes).to(tl.int8).reshape((2, 4))
            x0 = tl.join(x0.reshape((8,)), x0.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x1 = tl.join(x1.reshape((8,)), x1.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x2 = tl.join(x2.reshape((8,)), x2.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x3 = tl.join(x3.reshape((8,)), x3.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))

            partial01 = tl.sum(
                q0_low.to(tl.int32) * x0.to(tl.int32), axis=1
            )
            partial23 = tl.sum(
                q1_low.to(tl.int32) * x0.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q2_low.to(tl.int32) * x1.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q3_low.to(tl.int32) * x1.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q0_high.to(tl.int32) * x2.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q1_high.to(tl.int32) * x2.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q2_high.to(tl.int32) * x3.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q3_high.to(tl.int32) * x3.to(tl.int32), axis=1
            )

            partial = tl.join(partial01, partial23).permute(
                1, 0
            ).reshape((4, 2))
            dot_scaled16 = tl.sum(partial, axis=1)
            dot = dot_scaled16.to(tl.float32) * (1.0 / 16.0)
            combined_scale = lhs_scale * rhs_scale
            result += dot * combined_scale
            lhs_group_ptr += lhs_group_stride
            rhs_group_ptr += rhs_group_stride

        result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)
        tl.store(out_ptr + tile * 4 + output_lanes, result)


def pack_lhs(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    groups = x.numel() // 32
    packed = torch.empty((groups, 34), dtype=torch.uint8)
    packed[:, :2].view(torch.float16).copy_(scale.reshape(groups, 1))
    packed[:, 2:].view(torch.int8).copy_(x.reshape(groups, 32))
    return packed.reshape(-1)


def pack_rhs(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    groups = k // 32
    tiles = n // 4
    packed = torch.empty((tiles, groups, 72), dtype=torch.uint8)
    packed[:, :, :8].view(torch.float16).copy_(
        scale.reshape(groups, tiles, 4).permute(1, 0, 2)
    )
    for tile in range(tiles):
        for group in range(groups):
            q = weight[tile * 4 : tile * 4 + 4, group * 32 : group * 32 + 32]
            vectors = []
            for k_begin, outputs in ((0, (0, 1)), (0, (2, 3)),
                                     (8, (0, 1)), (8, (2, 3))):
                lo = q[list(outputs), k_begin : k_begin + 8].to(torch.int16) & 15
                hi = q[list(outputs), k_begin + 16 : k_begin + 24].to(torch.int16) & 15
                vectors.append((lo | (hi << 4)).to(torch.uint8).reshape(-1))
            packed[tile, group, 8:].copy_(torch.cat(vectors))
    return packed.reshape(-1)


def last_compiled(kernel):
    cache = next(iter(kernel.device_caches.values()))[0]
    return list(cache.values())[-1]


def audit_codegen(
    compiled, unroll: int
) -> tuple[str, str, int, int, int, int, int]:
    """Fail compilation if the production W4 microtile loses its ISA shape."""
    assembly = compiled.asm["asm"].lower()
    llir = compiled.asm["llir"].lower()
    sdot_count = assembly.count("sdot")
    addp_count = assembly.count("addp")
    ld1r_count = assembly.count("ld1r")
    stack_load_store = sum(
        "[sp" in line and line.lstrip().startswith(("ld", "st"))
        for line in assembly.splitlines()
    )
    external_calls = sum(
        " call " in line and "llvm." not in line for line in llir.splitlines()
    )
    expected_sdot = 8 * unroll
    expected_addp = unroll
    if sdot_count != expected_sdot:
        raise RuntimeError(
            f"W4 codegen regression: expected {expected_sdot} SDOT, got {sdot_count}"
        )
    if addp_count != expected_addp:
        raise RuntimeError(
            f"W4 codegen regression: expected {expected_addp} ADDP, got {addp_count}"
        )
    # Two LD1R instructions broadcast clamp bounds and each unrolled K32 body
    # broadcasts four activation K8 slices.  SDOT count alone does not protect
    # this schedule: LDR(D)+MOV lane duplication is correct but 13-17% slower
    # on CIX.
    expected_ld1r = 2 + 4 * unroll
    if ld1r_count != expected_ld1r:
        raise RuntimeError(
            "W4 codegen regression: expected "
            f"{expected_ld1r} LD1R, got {ld1r_count}"
        )
    if "scvtf" not in assembly or "#4" not in assembly:
        raise RuntimeError("W4 codegen regression: fixed-point SCVTF #4 is absent")
    if "smull" in assembly or "smlal" in assembly:
        raise RuntimeError("W4 codegen regression: widening multiply survived")
    if stack_load_store:
        raise RuntimeError(
            f"W4 codegen regression: {stack_load_store} stack load/store instructions"
        )
    if external_calls:
        raise RuntimeError(
            f"W4 codegen regression: {external_calls} non-LLVM external calls"
        )
    return (
        assembly,
        llir,
        sdot_count,
        addp_count,
        ld1r_count,
        stack_load_store,
        external_calls,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.k % 32 or args.n % 4:
        raise ValueError("requires K%32=0 and N%4=0")

    clamp = torch.tensor([-math.inf, math.inf], dtype=torch.float32)
    if args.compile_only:
        byte_buffer = torch.empty(1, dtype=torch.uint8)
        output = torch.empty(1, dtype=torch.float32)
        compiled = _kai_w4_layout_split_kernel.warmup(
            byte_buffer,
            byte_buffer,
            clamp,
            output,
            0,
            args.n // 4,
            K=args.k,
            N=args.n,
            UNROLL=args.unroll,
            grid=(1,),
        )
        (
            assembly,
            llir,
            sdot_count,
            addp_count,
            ld1r_count,
            stack_load_store,
            external_calls,
        ) = audit_codegen(compiled, args.unroll)
        print(
            f"COMPILED KAI-layout W4 K={args.k} N={args.n} "
            f"UNROLL={args.unroll}\n"
            f"asm_lines={len(assembly.splitlines())}\n"
            f"asm_sdot={sdot_count}\n"
            f"asm_addp={addp_count}\n"
            f"asm_ld1r={ld1r_count}\n"
            f"stack_load_store={stack_load_store}\n"
            f"external_calls={external_calls}"
        )
        return

    torch.manual_seed(4132)
    x = torch.randint(-127, 128, (args.k,), dtype=torch.int8)
    weight = torch.randint(-8, 8, (args.n, args.k), dtype=torch.int8)
    lhs_scale = (torch.rand(args.k // 32, dtype=torch.float16) * 0.02 + 0.001)
    rhs_scale = (
        torch.rand(args.k // 32, args.n, dtype=torch.float16) * 0.02 + 0.001
    )
    lhs = pack_lhs(x, lhs_scale)
    rhs = pack_rhs(weight, rhs_scale)
    output = torch.empty(args.n, dtype=torch.float32)

    _kai_w4_layout_split_kernel[(1,)](
        lhs,
        rhs,
        clamp,
        output,
        0,
        args.n // 4,
        K=args.k,
        N=args.n,
        UNROLL=args.unroll,
    )
    expected = torch.zeros(args.n, dtype=torch.float32)
    for group in range(args.k // 32):
        begin = group * 32
        dot = x[begin : begin + 32].to(torch.int32) @ weight[:, begin : begin + 32].to(torch.int32).T
        expected += dot.float() * lhs_scale[group].float() * rhs_scale[group].float()
    torch.testing.assert_close(output, expected, rtol=2.0e-6, atol=2.0e-5)

    compiled = last_compiled(_kai_w4_layout_split_kernel)
    (
        assembly,
        llir,
        sdot_count,
        addp_count,
        ld1r_count,
        stack_load_store,
        external_calls,
    ) = audit_codegen(compiled, args.unroll)
    print(
        f"PASS KAI-layout W4 K={args.k} N={args.n} UNROLL={args.unroll}\n"
        f"asm_lines={len(assembly.splitlines())}\n"
        f"asm_sdot={sdot_count}\n"
        f"asm_addp={addp_count}\n"
        f"asm_ld1r={ld1r_count}\n"
        f"stack_load_store={stack_load_store}\n"
        f"external_calls={external_calls}"
    )


if __name__ == "__main__":
    main()
