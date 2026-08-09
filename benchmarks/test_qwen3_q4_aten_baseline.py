#!/usr/bin/env python3
"""Validate the controlled ATen/KleidiAI Q4 Qwen benchmark route."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = Path(
    os.getenv(
        "TRITON_CPU_VENV_SITE",
        "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
    )
)
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from flag_gems.runtime.backend._arm.q4 import (  # noqa: E402
    AtenQ4Linear,
    optimize_qwen3_q4_aten,
    prepare_aten_grouped_weight,
    prepare_aten_weight,
)


def tiny_qwen_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=128,
        rms_norm_eps=1.0e-6,
        attention_bias=False,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
    )


def main() -> None:
    if not torch.backends.kleidiai.is_available():
        raise RuntimeError("this baseline requires ATen KleidiAI")
    torch.manual_seed(0)
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))

    weight = torch.randn((128, 128), dtype=torch.bfloat16)
    assert torch.equal(
        prepare_aten_weight(weight, chunk_rows=128),
        prepare_aten_weight(weight, chunk_rows=32),
    )
    linear = AtenQ4Linear.from_weight(weight)
    with torch.inference_mode():
        result = linear(torch.randn((3, 128), dtype=torch.bfloat16))
    assert result.shape == (3, 128)
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()

    signed = torch.randint(-8, 8, (128, 256), dtype=torch.int8)
    grouped_scale = torch.rand((128, 2), dtype=torch.bfloat16)
    grouped_packed = prepare_aten_grouped_weight(
        signed, grouped_scale, group_size=128
    )
    grouped = AtenQ4Linear.from_grouped_int4(
        signed, grouped_scale, group_size=128
    )
    assert torch.equal(grouped.packed_weight, grouped_packed)
    with torch.inference_mode():
        grouped_output = grouped(
            torch.randn((3, 256), dtype=torch.bfloat16)
        )
    assert grouped_output.shape == (3, 128)
    assert grouped_output.dtype == torch.bfloat16
    assert torch.isfinite(grouped_output).all()

    model = Qwen3ForCausalLM(tiny_qwen_config()).to(torch.bfloat16).eval()
    setup = optimize_qwen3_q4_aten(model)
    assert setup["quantization"] == "aten_kleidiai_dynamic_q4a8_k32"
    assert setup["logical_q4_linears"] == 8
    assert setup["physical_q4_matrices_total"] == 5
    ids = torch.tensor([[1, 7, 13, 29]], dtype=torch.long)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert output.logits.shape == (1, 4, 256)
    assert torch.isfinite(output.logits).all()
    print(json.dumps({"status": "PASS", "setup": setup}, default=str))


if __name__ == "__main__":
    main()
