#!/usr/bin/env python3
"""Validate MiniCPM production Q4/Q8 weight contracts and chunked packs."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    unpack_weights_i8mm_kai,
)
from flag_gems.runtime.backend._arm.q4.linear import (
    pack_rhs_qsi4c32p,
    pack_rhs_qsi4c32p_asym,
    pack_rhs_qsi8cxp,
    prepare_w8_weight,
    prepare_w8_weight_kai,
    prepare_weight,
    prepare_weight_asym,
    quantize_q4_0,
)


def unpack_decode_rhs(
    packed: torch.Tensor, n: int, k: int, block_n: int
) -> torch.Tensor:
    """Invert the N64-blocked SDOT pack to row-major ``[N,K]``."""
    blocked = packed.reshape(
        n // block_n, k // 4, block_n // 4, 4, 4
    )
    kmajor = blocked.permute(1, 0, 2, 3, 4).reshape(k // 4, n // 4, 4, 4)
    return kmajor.permute(0, 3, 1, 2).reshape(k, n).T.contiguous()


def reference_w8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = weight.float()
    scale = (values.abs().amax(dim=1) / 127.0).clamp_min(1.0e-8)
    quantized = (
        (values / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    )
    return quantized, scale


def unpack_qsi8cxp(
    packed: torch.Tensor, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    panels = packed.view(torch.uint8).reshape(n // 4, 4 * k + 48)
    values = (
        panels[:, : 4 * k]
        .view(torch.int8)
        .reshape(n // 4, k // 8, 4, 8)
        .permute(0, 2, 1, 3)
        .reshape(n, k)
        .contiguous()
    )
    sums = panels[:, 4 * k : 4 * k + 16].view(torch.int32).reshape(n)
    scales = (
        panels[:, 4 * k + 16 : 4 * k + 32]
        .view(torch.float32)
        .reshape(n)
    )
    bias = (
        panels[:, 4 * k + 32 : 4 * k + 48]
        .view(torch.float32)
        .reshape(n)
    )
    return values, sums, scales, bias


def check_kai_rhs_reference(
    library: Path, quantized: torch.Tensor, scale: torch.Tensor
) -> None:
    reference = ctypes.CDLL(str(library.resolve()))
    reference.kai_reference_qsi8cxp_size.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    reference.kai_reference_qsi8cxp_size.restype = ctypes.c_size_t
    reference.kai_reference_pack_qsi8cxp.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    n, k = quantized.shape
    packed = pack_rhs_qsi8cxp(quantized, scale)
    size = reference.kai_reference_qsi8cxp_size(n, k)
    if size != packed.numel():
        raise AssertionError(f"KAI RHS size mismatch: {size}/{packed.numel()}")
    expected = torch.empty((size,), dtype=torch.int8)
    reference.kai_reference_pack_qsi8cxp(
        n,
        k,
        quantized.data_ptr(),
        scale.data_ptr(),
        expected.data_ptr(),
    )
    if not torch.equal(packed, expected):
        mismatch = int((packed != expected).sum())
        raise AssertionError(f"qsi8cxp differs from KleidiAI: {mismatch} bytes")


def check_synthetic(kai_rhs_reference: Path | None = None) -> None:
    torch.manual_seed(20260809)
    weight = torch.randn((192, 128), dtype=torch.bfloat16) * 0.05
    weight[64] = 0

    direct_q4, direct_scale = quantize_q4_0(weight)
    direct_packed = pack_rhs_qsi4c32p(direct_q4, direct_scale)
    chunked_packed = prepare_weight(weight, chunk_rows=64)
    if not torch.equal(direct_packed, chunked_packed):
        raise AssertionError("chunked G32 Q4 pack differs from direct pack")
    direct_asym = pack_rhs_qsi4c32p_asym(
        direct_q4, direct_scale.to(torch.bfloat16)
    )
    if not torch.equal(direct_asym, prepare_weight_asym(weight, chunk_rows=64)):
        raise AssertionError("chunked asymmetric G32 pack differs from direct pack")

    expected_q, expected_scale = reference_w8(weight)
    decode, prefill, scale = prepare_w8_weight(weight, chunk_rows=64)
    if not torch.equal(scale, expected_scale):
        raise AssertionError("W8 per-channel scales differ from reference")
    if not torch.equal(unpack_decode_rhs(decode, 192, 128, 64), expected_q):
        raise AssertionError("W8 SDOT pack does not preserve quantized values")
    unpacked_prefill = unpack_weights_i8mm_kai(prefill, 192, 128).T
    if not torch.equal(unpacked_prefill, expected_q):
        raise AssertionError("W8 I8MM pack does not preserve quantized values")

    direct_kai = pack_rhs_qsi8cxp(expected_q, expected_scale)
    if kai_rhs_reference is not None:
        check_kai_rhs_reference(
            kai_rhs_reference, expected_q, expected_scale
        )
    chunked_kai = prepare_w8_weight_kai(weight, chunk_rows=64)
    if not torch.equal(direct_kai, chunked_kai):
        raise AssertionError("chunked exact-KAI W8 pack differs from direct pack")
    kai_q, kai_sum, kai_scale, kai_bias = unpack_qsi8cxp(
        chunked_kai, 192, 128
    )
    if not torch.equal(kai_q, expected_q):
        raise AssertionError("qsi8cxp pack does not preserve W8 values")
    if not torch.equal(kai_sum, expected_q.to(torch.int32).sum(dim=1)):
        raise AssertionError("qsi8cxp RHS correction sums are incorrect")
    if not torch.equal(kai_scale, expected_scale):
        raise AssertionError("qsi8cxp scales differ from W8 reference")
    if torch.count_nonzero(kai_bias):
        raise AssertionError("bias-free qsi8cxp pack contains non-zero bias")


def check_checkpoint(source: Path, w8_model: Path) -> None:
    from safetensors import safe_open

    source_state = torch.load(
        source / "pytorch_model.bin",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    keys = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    )
    with safe_open(
        w8_model / "model.safetensors", framework="pt", device="cpu"
    ) as checkpoint:
        for weight_key in keys:
            scale_key = weight_key + "_scale"
            expected_q = checkpoint.get_tensor(weight_key)
            expected_scale = checkpoint.get_tensor(scale_key).reshape(-1)
            actual_q, actual_scale = reference_w8(source_state[weight_key])
            if not torch.equal(actual_scale, expected_scale):
                raise AssertionError(f"checkpoint scale mismatch: {weight_key}")
            if not torch.equal(actual_q, expected_q):
                mismatches = int((actual_q != expected_q).sum())
                raise AssertionError(
                    f"checkpoint INT8 mismatch: {weight_key}: {mismatches}"
                )

        lm_head = checkpoint.get_slice("lm_head.weight")
        n, k = lm_head.get_shape()
        if n % 64 or k % 32:
            raise AssertionError(
                f"MiniCPM lm_head cannot use W8 SDOT/I8MM: {(n, k)}"
            )


def check_all_linear_w8_checkpoint(source: Path, derived: Path) -> None:
    """Validate the guide's offline W8 lm_head derivation chunk by chunk."""
    from safetensors import safe_open

    config = json.loads((derived / "config.json").read_text())
    quantization = config["quantization_config"]
    if quantization.get("ignore"):
        raise AssertionError("all-Linear W8 checkpoint still ignores lm_head")
    targets = {
        target
        for group in quantization["config_groups"].values()
        for target in group.get("targets", [])
    }
    if "lm_head" not in targets:
        raise AssertionError("all-Linear W8 config does not target lm_head")

    with (
        safe_open(
            source / "model.safetensors", framework="pt", device="cpu"
        ) as source_file,
        safe_open(
            derived / "model.safetensors", framework="pt", device="cpu"
        ) as derived_file,
    ):
        for key in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight_scale",
        ):
            if not torch.equal(
                source_file.get_tensor(key), derived_file.get_tensor(key)
            ):
                raise AssertionError(f"offline derivation changed body tensor {key}")
        source_head = source_file.get_slice("lm_head.weight")
        actual_q = derived_file.get_slice("lm_head.weight")
        actual_scale = derived_file.get_slice("lm_head.weight_scale")
        n, k = source_head.get_shape()
        if actual_q.get_shape() != [n, k] or actual_scale.get_shape() != [n, 1]:
            raise AssertionError("offline W8 lm_head has unexpected shape")
        for row_begin in range(0, n, 512):
            row_end = min(row_begin + 512, n)
            values = source_head[row_begin:row_end].to(torch.float32)
            scale = (values.abs().amax(dim=1) / 127.0).clamp_min_(1.0e-8)
            quantized = (
                (values / scale[:, None])
                .round()
                .clamp_(-127, 127)
                .to(torch.int8)
            )
            if not torch.equal(actual_q[row_begin:row_end], quantized):
                raise AssertionError(f"offline W8 lm_head mismatch at row {row_begin}")
            if not torch.equal(
                actual_scale[row_begin:row_end].reshape(-1), scale
            ):
                raise AssertionError(
                    f"offline W8 lm_head scale mismatch at row {row_begin}"
                )


def check_q4_head_g32_checkpoint(source: Path, derived: Path) -> None:
    """Validate the guide's offline G32 head while preserving G128 body."""
    from safetensors import safe_open

    config = json.loads((derived / "config.json").read_text())
    quantization = config["quantization_config"]
    if quantization.get("ignore"):
        raise AssertionError("G32 checkpoint still ignores lm_head")
    head_groups = [
        group
        for group in quantization["config_groups"].values()
        if "lm_head" in group.get("targets", [])
    ]
    if len(head_groups) != 1 or head_groups[0]["weights"]["group_size"] != 32:
        raise AssertionError("derived Q4 config does not select G32 lm_head")

    with (
        safe_open(
            source / "model.safetensors", framework="pt", device="cpu"
        ) as source_file,
        safe_open(
            derived / "model.safetensors", framework="pt", device="cpu"
        ) as derived_file,
    ):
        for key in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight_scale",
        ):
            if not torch.equal(
                source_file.get_tensor(key), derived_file.get_tensor(key)
            ):
                raise AssertionError(f"G32 derivation changed body tensor {key}")
        source_head = source_file.get_slice("lm_head.weight")
        actual_q = derived_file.get_slice("lm_head.weight")
        actual_scale = derived_file.get_slice("lm_head.weight_scale")
        n, k = source_head.get_shape()
        if actual_q.get_shape() != [n, k] or actual_scale.get_shape() != [
            n,
            k // 32,
        ]:
            raise AssertionError("offline G32 lm_head has unexpected shape")
        for row_begin in range(0, n, 512):
            row_end = min(row_begin + 512, n)
            blocks = source_head[row_begin:row_end].to(torch.float32).reshape(
                row_end - row_begin, k // 32, 32
            )
            indices = blocks.abs().argmax(dim=-1, keepdim=True)
            scale = torch.gather(blocks, -1, indices) / -8.0
            reciprocal = torch.where(
                scale != 0.0, 1.0 / scale, torch.zeros_like(scale)
            )
            quantized = (
                (blocks * reciprocal)
                .round()
                .clamp_(-8, 7)
                .to(torch.int8)
                .reshape(row_end - row_begin, k)
            )
            if not torch.equal(actual_q[row_begin:row_end], quantized):
                raise AssertionError(f"offline G32 head mismatch at row {row_begin}")
            if not torch.equal(
                actual_scale[row_begin:row_end], scale.squeeze(-1).to(torch.bfloat16)
            ):
                raise AssertionError(
                    f"offline G32 head scale mismatch at row {row_begin}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/home/cix/"
            "MiniCPM5-2.6B-0426_job_327123_step_24000_fusion_think_512k"
        ),
    )
    parser.add_argument(
        "--w8-model",
        type=Path,
        default=Path("/home/cix/MiniCPM5-2.6B-W8A8-CT"),
    )
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument(
        "--kai-rhs-reference",
        type=Path,
        help="shared KleidiAI RHS pack reference for byte-exact validation",
    )
    parser.add_argument(
        "--all-linear-w8-model",
        type=Path,
        help="derived checkpoint with an offline per-channel W8 lm_head",
    )
    parser.add_argument(
        "--q4-model",
        type=Path,
        help="source G128-body checkpoint with a BF16 lm_head",
    )
    parser.add_argument(
        "--head-g32-q4-model",
        type=Path,
        help="derived G128-body/G32-head Q4 checkpoint",
    )
    args = parser.parse_args()

    check_synthetic(args.kai_rhs_reference)
    if not args.skip_checkpoint:
        check_checkpoint(args.source.resolve(), args.w8_model.resolve())
        if args.all_linear_w8_model is not None:
            check_all_linear_w8_checkpoint(
                args.w8_model.resolve(), args.all_linear_w8_model.resolve()
            )
        if (args.q4_model is None) != (args.head_g32_q4_model is None):
            raise ValueError(
                "--q4-model and --head-g32-q4-model require each other"
            )
        if args.q4_model is not None:
            check_q4_head_g32_checkpoint(
                args.q4_model.resolve(), args.head_g32_q4_model.resolve()
            )
    print(
        "PASS MiniCPM production formats: Q4 body=G128 checkpoint, "
        "Q4 head=G32; Q8 body/head=qai8dxp x per-channel qsi8cxp; "
        "chunked packs exact"
    )


if __name__ == "__main__":
    main()
