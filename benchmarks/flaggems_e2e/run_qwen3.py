#!/usr/bin/env python3
"""Reproducible Qwen3 CPU end-to-end benchmark for FlagGems/FlagTree.

Run this file with ``python -S``.  The explicit paths below avoid the editable
FlagGems/FlagTree installs in the development virtualenv silently redirecting
imports to a different checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/home/kevin/triton-opt-cpu")
VENV_SITE = Path("/home/cix/venv-fep-e2e/lib/python3.11/site-packages")
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON",
        ROOT / "ports/triton-cpu-3.7.2/python",
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
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


MODES = (
    "eager",
    "enable",
    "arm",
    "int8",
    "int8-qwen3",
    "int8-qwen3-argmax",
    "int8-qwen3-attn",
    "int8-qwen3-attn-argmax",
    "int8-qwen3-qkv-argmax",
    "int8-qwen3-qkv-attn-argmax",
    "q4-qwen3",
    "q4-qwen3-no-lm-head",
    "q4-aten",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model", default="/home/cix/models/Qwen3-0.6B")
    parser.add_argument("--prompt", default="The future of edge AI is")
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="Repeat/truncate the tokenized prompt to this exact length.",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefill-repeats", type=int, default=5)
    parser.add_argument(
        "--decode-repeats",
        type=int,
        default=0,
        help=(
            "time isolated one-token decode steps; each sample builds a "
            "fresh KV cache outside the timed region"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("none", "prefill", "decode", "generate"),
        default="none",
    )
    parser.add_argument("--profile-rows", type=int, default=40)
    parser.add_argument(
        "--disable-qk-norm-fusion",
        action="store_true",
        help="keep independent Q and K RMSNorm calls for controlled A/B",
    )
    parser.add_argument(
        "--disable-qk-qkv-fusion",
        action="store_true",
        help="keep the joined Q/K RMSNorm as a separate Triton launch",
    )
    parser.add_argument(
        "--disable-decode-attention-fastpath",
        action="store_true",
        help="keep standard Q/K/V split/view/transpose module dispatches",
    )
    parser.add_argument("--json-out")
    return parser.parse_args()


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def apply_mode(
    mode: str,
    model: torch.nn.Module,
    *,
    enable_qk_norm_fusion: bool = True,
) -> dict:
    setup: dict[str, object] = {}
    if mode == "eager":
        return setup

    import flag_gems

    setup["flaggems"] = flag_gems.__file__

    if mode in ("enable", "arm"):
        flag_gems.enable()
        setup["registered_ops"] = len(flag_gems.all_registered_ops())

    if mode == "arm":
        from flag_gems.runtime.backend._arm.ops import apply_arm_overrides

        # Apply the non-quantized, model-visible overrides.  The standard
        # enable() registrar already handles the regular aten op table.
        selected = [
            "argmax",
            "mm",
            "scaled_dot_product_attention",
            "fused_add_rms_norm",
            "apply_rotary_pos_emb",
            "silu_and_mul",
        ]
        apply_arm_overrides(include=selected)
        setup["arm_overrides"] = selected

    if mode == "int8":
        from flag_gems.runtime.backend._arm.int8 import quantize_and_replace_linears

        setup["int8_linears"] = quantize_and_replace_linears(model)

    if mode in (
        "int8-qwen3",
        "int8-qwen3-argmax",
        "int8-qwen3-attn",
        "int8-qwen3-attn-argmax",
        "int8-qwen3-qkv-argmax",
        "int8-qwen3-qkv-attn-argmax",
    ):
        from flag_gems.runtime.backend._arm.int8 import optimize_qwen3_w8a8

        setup.update(
            optimize_qwen3_w8a8(
                model,
                enable_attention=mode
                in (
                    "int8-qwen3-attn",
                    "int8-qwen3-attn-argmax",
                    "int8-qwen3-qkv-attn-argmax",
                ),
                enable_argmax=mode
                in (
                    "int8-qwen3-argmax",
                    "int8-qwen3-attn-argmax",
                    "int8-qwen3-qkv-argmax",
                    "int8-qwen3-qkv-attn-argmax",
                ),
                enable_qkv_fusion=mode
                in (
                    "int8-qwen3-qkv-argmax",
                    "int8-qwen3-qkv-attn-argmax",
                ),
                enable_qk_norm_fusion=enable_qk_norm_fusion,
            )
        )

    if mode in ("q4-qwen3", "q4-qwen3-no-lm-head"):
        from flag_gems.runtime.backend._arm.q4 import optimize_qwen3_q4

        setup.update(
            optimize_qwen3_q4(
                model,
                quantize_lm_head=mode == "q4-qwen3",
                # The ordinary-Triton embedding implementation is retained
                # for coverage experiments, but CIX microbenchmarks show the
                # optimized ATen gather is faster for 1--128 prompt tokens.
                enable_embedding=False,
                enable_attention=True,
                enable_argmax=True,
                enable_qk_norm_fusion=enable_qk_norm_fusion,
            )
        )

    if mode == "q4-aten":
        from flag_gems.runtime.backend._arm.q4 import optimize_qwen3_q4_aten

        setup.update(
            optimize_qwen3_q4_aten(
                model,
                quantize_lm_head=True,
                enable_embedding=False,
                enable_attention=True,
                enable_argmax=True,
                enable_qk_norm_fusion=enable_qk_norm_fusion,
            )
        )

    return setup


def generate(model, ids, attention_mask, tokenizer, new_tokens: int):
    return model.generate(
        ids,
        attention_mask=attention_mask,
        min_new_tokens=new_tokens,
        max_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )


def _make_decode_inputs(model, ids, attention_mask) -> dict[str, object]:
    prefill = model(
        input_ids=ids, attention_mask=attention_mask, use_cache=True
    )
    next_id = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_mask = torch.ones(
        (ids.shape[0], ids.shape[1] + 1), dtype=attention_mask.dtype
    )
    return {
        "input_ids": next_id,
        "attention_mask": decode_mask,
        "past_key_values": prefill.past_key_values,
        "cache_position": torch.tensor([ids.shape[1]], dtype=torch.long),
        "use_cache": True,
    }


def isolated_decode_ms(model, ids, attention_mask, repeats: int) -> list[float]:
    """Measure a warmed real decode step without folding in prefill/JIT."""
    # max_new_tokens=1 performs only the initial prefill forward, so generation
    # is not a decode warmup.  Compile the exact T=1/cache-length specialization
    # once before collecting samples.
    warm_inputs = _make_decode_inputs(model, ids, attention_mask)
    warm_output = model(**warm_inputs)
    warm_output.logits[:, -1, :].argmax(dim=-1)

    samples = []
    for _ in range(repeats):
        decode_inputs = _make_decode_inputs(model, ids, attention_mask)
        begin = time.perf_counter_ns()
        decoded = model(**decode_inputs)
        decoded.logits[:, -1, :].argmax(dim=-1)
        samples.append((time.perf_counter_ns() - begin) / 1e6)
    return samples


def profile_once(args, model, ids, attention_mask, tokenizer):
    if args.profile == "none":
        return None
    from torch.profiler import ProfilerActivity, profile

    decode_inputs = None
    if args.profile == "decode":
        with torch.inference_mode():
            decode_inputs = _make_decode_inputs(
                model, ids, attention_mask
            )

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.inference_mode():
            if args.profile == "prefill":
                model(
                    input_ids=ids, attention_mask=attention_mask, use_cache=True
                )
            elif args.profile == "decode":
                decoded = model(**decode_inputs)
                # Include the greedy vocabulary reduction: it is part of one
                # real generate step and is an explicit Triton replacement in
                # the argmax modes.
                decoded.logits[:, -1, :].argmax(dim=-1)
            else:
                generate(model, ids, attention_mask, tokenizer, 2)
    events = list(prof.key_averages())
    total_self_us = sum(event.self_cpu_time_total for event in events)
    triton_events = [
        event for event in events if event.key.startswith("triton::")
    ]
    triton_self_us = sum(
        event.self_cpu_time_total for event in triton_events
    )
    summary = {
        "scope": args.profile,
        "total_self_cpu_us": total_self_us,
        "triton_self_cpu_us": triton_self_us,
        "triton_coverage": (
            triton_self_us / total_self_us if total_self_us else 0.0
        ),
        "triton_ranges": {
            event.key: {
                "self_cpu_us": event.self_cpu_time_total,
                "calls": event.count,
                "share": (
                    event.self_cpu_time_total / total_self_us
                    if total_self_us
                    else 0.0
                ),
            }
            for event in sorted(
                triton_events,
                key=lambda item: item.self_cpu_time_total,
                reverse=True,
            )
        },
        "top_non_triton": [
            {
                "name": event.key,
                "self_cpu_us": event.self_cpu_time_total,
                "calls": event.count,
                "share": (
                    event.self_cpu_time_total / total_self_us
                    if total_self_us
                    else 0.0
                ),
            }
            for event in sorted(
                (
                    event
                    for event in events
                    if not event.key.startswith("triton::")
                ),
                key=lambda item: item.self_cpu_time_total,
                reverse=True,
            )[:20]
        ],
    }
    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cpu_time_total", row_limit=args.profile_rows
    )
    return table, summary


def main() -> None:
    args = parse_args()
    # profile_range.py snapshots this switch when the Arm backend is first
    # imported by apply_mode().  Enable it before that import so a requested
    # profiler run cannot silently report zero Triton coverage.
    if args.profile != "none":
        os.environ["FLAGGEMS_PROFILE_RANGES"] = "1"
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    # Inter-op is process-global and must be set before parallel work starts.
    torch.set_num_interop_threads(1)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    load_s = time.perf_counter() - started

    setup_started = time.perf_counter()
    setup = apply_mode(
        args.mode,
        model,
        enable_qk_norm_fusion=not args.disable_qk_norm_fusion,
    )
    if args.mode in ("q4-qwen3", "q4-qwen3-no-lm-head"):
        if args.disable_qk_qkv_fusion:
            from flag_gems.runtime.backend._arm.q4 import (
                set_fused_qk_norm_qkv_enabled,
            )

            set_fused_qk_norm_qkv_enabled(False)
        if args.disable_decode_attention_fastpath:
            from flag_gems.runtime.backend._arm.q4 import (
                set_decode_attention_fastpath_enabled,
            )

            set_decode_attention_fastpath_enabled(False)
    setup_s = time.perf_counter() - setup_started
    encoded = tokenizer(args.prompt, return_tensors="pt")
    ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    if args.prompt_tokens:
        repeats = (args.prompt_tokens + ids.shape[1] - 1) // ids.shape[1]
        ids = ids.repeat(1, repeats)[:, : args.prompt_tokens].contiguous()
        attention_mask = torch.ones_like(ids)

    # Warm both prefill and decode/JIT paths outside the measurement.
    with torch.inference_mode():
        if args.warmup_tokens:
            generate(model, ids, attention_mask, tokenizer, args.warmup_tokens)

    prefill_ms = []
    with torch.inference_mode():
        for _ in range(args.prefill_repeats):
            t0 = time.perf_counter_ns()
            model(input_ids=ids, attention_mask=attention_mask, use_cache=True)
            prefill_ms.append((time.perf_counter_ns() - t0) / 1e6)

    generation_s = []
    output = None
    with torch.inference_mode():
        for _ in range(args.repeats):
            t0 = time.perf_counter_ns()
            output = generate(
                model, ids, attention_mask, tokenizer, args.new_tokens
            )
            generation_s.append((time.perf_counter_ns() - t0) / 1e9)

    decode_ms = []
    if args.decode_repeats:
        with torch.inference_mode():
            decode_ms = isolated_decode_ms(
                model, ids, attention_mask, args.decode_repeats
            )

    assert output is not None
    generated_ids = output[0, ids.shape[1] :]
    result = {
        "mode": args.mode,
        "model": args.model,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "prompt_tokens": int(ids.shape[1]),
        "new_tokens": args.new_tokens,
        "threads": args.threads,
        "torch": torch.__version__,
        "load_s": load_s,
        "setup_s": setup_s,
        "rss_mib": rss_mib(),
        "prefill_ms": prefill_ms,
        "prefill_median_ms": statistics.median(prefill_ms),
        "generation_s": generation_s,
        "generation_median_s": statistics.median(generation_s),
        "tokens_per_s": args.new_tokens / statistics.median(generation_s),
        "generated_token_ids": generated_ids.tolist(),
        "text": tokenizer.decode(output[0], skip_special_tokens=True),
        "setup": setup,
    }
    if decode_ms:
        result["decode_step_ms"] = decode_ms
        result["decode_step_median_ms"] = statistics.median(decode_ms)

    try:
        import triton

        result["triton"] = triton.__file__
        result["target"] = str(triton.runtime.driver.active.get_current_target())
    except ImportError:
        result["triton"] = None

    if args.mode in ("q4-qwen3", "q4-qwen3-no-lm-head"):
        from flag_gems.runtime.backend._arm.q4 import stats as q4_stats

        result["q4_stats"] = q4_stats()

    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
    profile_result = profile_once(args, model, ids, attention_mask, tokenizer)
    if profile_result:
        table, profile_summary = profile_result
        result["profile"] = profile_summary
        print(
            "PROFILE_JSON="
            + json.dumps(profile_summary, ensure_ascii=False, sort_keys=True)
        )
        print("\nPROFILE_TABLE")
        print(table)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
