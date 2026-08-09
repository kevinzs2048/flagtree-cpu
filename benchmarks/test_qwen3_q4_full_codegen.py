#!/usr/bin/env python3
"""Correctness and routing checks for the complete Qwen3 Q4 CPU path."""

from __future__ import annotations

import json
import importlib
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
os.environ.setdefault("FLAGGEMS_ARM_ATTN_DISABLE_RUNTIME", "1")
os.environ.setdefault("FLAGGEMS_ARM_ATTN_SHORT_PREFILL_CODEGEN", "1")
os.environ.setdefault("FLAGGEMS_ARM_ROPE_FREQUENCY_CODEGEN", "1")

import torch  # noqa: E402
import triton  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from flag_gems.runtime.backend._arm.ops.attention import (  # noqa: E402
    _aten_sdpa,
    _triton_flash_attn_short_prefill,
)
from flag_gems.runtime.backend._arm.q4 import (  # noqa: E402
    TritonEmbedding,
    optimize_qwen3_q4,
    prepare_weight,
    stats,
)

rope_patch_module = importlib.import_module(  # noqa: E402
    "flag_gems.runtime.backend._arm.fused.patch_qwen3_rope"
)


def require_expected_triton() -> None:
    expected = (TRITON_PYTHON / "triton").resolve()
    actual = Path(triton.__file__).resolve()
    if expected not in actual.parents:
        raise RuntimeError(
            f"wrong Triton source: expected under {expected}, got {actual}"
        )


def check_chunked_weight_pack() -> None:
    weight = torch.randn((64, 64), dtype=torch.bfloat16)
    monolithic = prepare_weight(weight, chunk_rows=64)
    chunked = prepare_weight(weight, chunk_rows=16)
    assert torch.equal(chunked, monolithic)


def check_embedding() -> None:
    embedding = torch.nn.Embedding(256, 128, dtype=torch.bfloat16).eval()
    triton_embedding = TritonEmbedding(embedding)
    for shape in ((1, 1), (1, 12), (2, 7)):
        indices = torch.randint(0, 256, shape)
        assert torch.equal(triton_embedding(indices), embedding(indices))


def check_prefill_rope() -> None:
    from transformers.models.qwen3 import modeling_qwen3 as qwen3

    if not rope_patch_module._PATCHED:
        rope_patch_module.patch_qwen3_rope()
    original = rope_patch_module._PATCHED["original"]
    original_frequency = rope_patch_module._ROTARY_PATCHED["original"]
    torch.manual_seed(11)
    q = torch.randn((1, 4, 7, 128), dtype=torch.bfloat16)
    k = torch.randn((1, 2, 7, 128), dtype=torch.bfloat16)
    angles = torch.randn((1, 7, 64), dtype=torch.float32)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(torch.bfloat16)
    expected_q, expected_k = original(q, k, cos, sin)
    actual_q, actual_k = qwen3.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert actual_q.is_contiguous() and actual_k.is_contiguous()

    rotary = qwen3.Qwen3RotaryEmbedding(tiny_qwen_config())
    hidden = torch.empty((1, 7, 128), dtype=torch.bfloat16)
    positions = torch.arange(7).reshape(1, -1)
    expected_cos, expected_sin = original_frequency(
        rotary, hidden, positions
    )
    actual_cos, actual_sin = rotary(hidden, positions)
    assert torch.equal(actual_cos, expected_cos)
    assert torch.equal(actual_sin, expected_sin)
    assert rope_patch_module.unpatch_qwen3_rope() == 2


def check_short_prefill_attention() -> float:
    torch.manual_seed(23)
    q = torch.randn((1, 4, 12, 128), dtype=torch.bfloat16)
    k = torch.randn((1, 2, 12, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    expected = _aten_sdpa(q, k, v, is_causal=True, enable_gqa=True)
    actual = _triton_flash_attn_short_prefill(
        q, k, v, 128**-0.5, True
    )
    max_abs = float((actual.float() - expected.float()).abs().max())
    assert torch.isfinite(actual).all()
    assert max_abs <= 0.015625, max_abs
    return max_abs


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


def check_complete_model() -> dict[str, object]:
    model = Qwen3ForCausalLM(tiny_qwen_config()).to(torch.bfloat16).eval()
    setup = optimize_qwen3_q4(
        model,
        quantize_lm_head=True,
        enable_embedding=True,
        enable_attention=True,
        enable_argmax=True,
        enable_qk_norm_fusion=True,
    )
    assert setup["logical_q4_linears"] == 8
    assert setup["physical_layer_matrices"] == 4
    assert setup["lm_head_q4"] is True
    assert setup["embedding_triton"] is True
    assert setup["input_rmsnorm_qkv_fusion"] is True
    assert setup["post_add_rmsnorm_gateup_fusion"] is True
    assert setup["qk_norm_qkv_fusion"] is True
    assert setup["decode_attention_fastpath"] is True

    ids = torch.tensor([[1, 7, 13, 29]], dtype=torch.long)
    mask = torch.ones_like(ids)
    before = stats()
    with torch.inference_mode():
        prefill = model(input_ids=ids, attention_mask=mask, use_cache=True)
        assert prefill.logits.shape == (1, 4, 256)
        next_id = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        decoded = model(
            input_ids=next_id,
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            past_key_values=prefill.past_key_values,
            cache_position=torch.tensor([4]),
            use_cache=True,
        )
        token = decoded.logits[:, -1].argmax(dim=-1)
    assert decoded.logits.shape == (1, 1, 256)
    assert torch.isfinite(decoded.logits).all()
    assert token.shape == (1,)
    after = stats()
    assert after["codegen_prefill_calls"] - before["codegen_prefill_calls"] == 5
    assert after["decode_codegen_calls"] - before["decode_codegen_calls"] == 5
    assert (
        after["fused_rmsnorm_decode_calls"]
        - before["fused_rmsnorm_decode_calls"]
        == 1
    )
    assert (
        after["fused_add_rmsnorm_decode_calls"]
        - before["fused_add_rmsnorm_decode_calls"]
        == 1
    )
    assert (
        after["fused_rmsnorm_qk_norm_decode_calls"]
        - before["fused_rmsnorm_qk_norm_decode_calls"]
        == 1
    )
    assert model.model.layers[0].self_attn._triton_q4_decode_fastpath_calls == 1
    return setup


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))
    require_expected_triton()
    check_chunked_weight_pack()
    check_embedding()
    check_prefill_rope()
    attention_max_abs = check_short_prefill_attention()
    setup = check_complete_model()
    print(
        json.dumps(
            {
                "status": "PASS",
                "triton": triton.__file__,
                "attention_max_abs": attention_max_abs,
                "setup": setup,
                "q4_stats": stats(),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
