#!/usr/bin/env python3
"""Derive the deployment G32 Q4 lm_head from a G128-body checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def quantize_head_g32(
    source_slice,
    n: int,
    k: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k % 32:
        raise ValueError("G32 lm_head requires K%32=0")
    quantized = torch.empty((n, k), dtype=torch.int8)
    scales = torch.empty((n, k // 32), dtype=torch.bfloat16)
    for row_begin in range(0, n, chunk_rows):
        row_end = min(row_begin + chunk_rows, n)
        blocks = source_slice[row_begin:row_end].to(torch.float32).reshape(
            row_end - row_begin, k // 32, 32
        )
        if not torch.isfinite(blocks).all():
            raise ValueError("lm_head contains a non-finite value")
        max_indices = blocks.abs().argmax(dim=-1, keepdim=True)
        scale = torch.gather(blocks, -1, max_indices) / -8.0
        reciprocal = torch.where(
            scale != 0.0, 1.0 / scale, torch.zeros_like(scale)
        )
        quantized[row_begin:row_end].copy_(
            (blocks * reciprocal)
            .round()
            .clamp_(-8, 7)
            .to(torch.int8)
            .reshape(row_end - row_begin, k)
        )
        scales[row_begin:row_end].copy_(scale.squeeze(-1))
    return quantized, scales


def head_g32_config(source_config: dict) -> dict:
    config = deepcopy(source_config)
    quantization = config.get("quantization_config")
    groups = quantization.get("config_groups") if quantization else None
    if not isinstance(groups, dict) or not groups:
        raise ValueError("source checkpoint has no compressed-tensors group")
    template = deepcopy(next(iter(groups.values())))
    weights = template.get("weights", {})
    if weights.get("num_bits") != 4 or weights.get("group_size") != 128:
        raise ValueError("source body is not G128 Q4")
    body = deepcopy(template)
    body["targets"] = ["Linear"]
    head = deepcopy(template)
    head["targets"] = ["lm_head"]
    head["weights"]["group_size"] = 32
    quantization["config_groups"] = {
        "group_body_g128": body,
        "group_lm_head_g32": head,
    }
    quantization["ignore"] = []
    return config


def derive(source: Path, output: Path, chunk_rows: int) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    source_model = source / "model.safetensors"
    source_config_path = source / "config.json"
    if not source_model.is_file() or not source_config_path.is_file():
        raise FileNotFoundError("source requires config.json and model.safetensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        with safe_open(source_model, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if "lm_head.weight_scale" in keys:
                raise ValueError("source lm_head is already quantized")
            head_slice = handle.get_slice("lm_head.weight")
            n, k = head_slice.get_shape()
            quantized, scales = quantize_head_g32(
                head_slice, n, k, chunk_rows
            )
            tensors = {key: handle.get_tensor(key) for key in keys}
            tensors["lm_head.weight"] = quantized
            tensors["lm_head.weight_scale"] = scales
            save_file(
                tensors,
                temporary / "model.safetensors",
                metadata=handle.metadata(),
            )

        config = head_g32_config(json.loads(source_config_path.read_text()))
        (temporary / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )
        for source_file in source.iterdir():
            if not source_file.is_file() or source_file.name in {
                "config.json",
                "model.safetensors",
            }:
                continue
            shutil.copy2(source_file, temporary / source_file.name)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=512)
    args = parser.parse_args()
    derive(args.source.resolve(), args.output.resolve(), args.chunk_rows)
    print(f"wrote G128-body/G32-head checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
