#!/usr/bin/env python3
"""Derive a vLLM-loadable all-Linear W8 checkpoint for MiniCPM5.

The supplied compressed-tensors checkpoint leaves ``lm_head`` in BF16.  The
deployment guide requires per-channel INT8 for every Linear, including the
head.  This tool preserves the existing body tensors byte-for-byte, quantizes
only ``lm_head`` offline, and publishes the result atomically.
"""

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


def quantize_rows(
    source_slice,
    n: int,
    k: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantized = torch.empty((n, k), dtype=torch.int8)
    scales = torch.empty((n, 1), dtype=torch.float32)
    for row_begin in range(0, n, chunk_rows):
        row_end = min(row_begin + chunk_rows, n)
        values = source_slice[row_begin:row_end].to(torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("lm_head contains a non-finite value")
        scale = (values.abs().amax(dim=1) / 127.0).clamp_min_(1.0e-8)
        quantized[row_begin:row_end].copy_(
            (values / scale[:, None])
            .round()
            .clamp_(-127, 127)
            .to(torch.int8)
        )
        scales[row_begin:row_end, 0].copy_(scale)
    return quantized, scales


def all_linear_config(source_config: dict) -> dict:
    config = deepcopy(source_config)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("source checkpoint has no quantization_config")
    groups = quantization.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("source checkpoint has no compressed-tensors group")
    template = deepcopy(next(iter(groups.values())))
    weights = template.get("weights", {})
    if weights.get("num_bits") != 8 or weights.get("strategy") != "channel":
        raise ValueError("source checkpoint is not per-channel W8")
    body = deepcopy(template)
    body["targets"] = ["Linear"]
    head = deepcopy(template)
    head["targets"] = ["lm_head"]
    quantization["config_groups"] = {
        "group_body_w8": body,
        "group_lm_head_w8": head,
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
            if "lm_head.weight" not in keys:
                raise ValueError("source checkpoint has no lm_head.weight")
            head_slice = handle.get_slice("lm_head.weight")
            n, k = head_slice.get_shape()
            quantized, scales = quantize_rows(
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

        config = all_linear_config(json.loads(source_config_path.read_text()))
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

        with safe_open(
            temporary / "model.safetensors", framework="pt", device="cpu"
        ) as result:
            if not torch.equal(result.get_tensor("lm_head.weight"), quantized):
                raise AssertionError("published lm_head INT8 differs")
            if not torch.equal(
                result.get_tensor("lm_head.weight_scale"), scales
            ):
                raise AssertionError("published lm_head scales differ")
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
    print(f"wrote all-Linear W8 checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
