#!/usr/bin/env python3
"""Paired end-to-end decode A/B for the compiler-visible Q4 fusion.

The model and packed weights are created once.  Legacy ``pack -> GEMV`` and
single-entry ``pack + GEMV`` routes then alternate inside the same process so
board temperature, allocator state, and weight preparation cannot masquerade
as a codegen improvement.  Run with ``python -S`` and pin the intended cores.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from run_qwen3 import (  # establishes the explicit source import paths
    AutoModelForCausalLM,
    AutoTokenizer,
    _make_decode_inputs,
    apply_mode,
    torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/cix/models/Qwen3-1.7B")
    parser.add_argument("--prompt", default="The future of edge AI is")
    parser.add_argument("--prompt-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=15)
    parser.add_argument("--json-out")
    return parser.parse_args()


def make_prompt(tokenizer, text: str, prompt_tokens: int):
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded.input_ids
    if prompt_tokens:
        repeats = (prompt_tokens + ids.shape[1] - 1) // ids.shape[1]
        ids = ids.repeat(1, repeats)[:, :prompt_tokens].contiguous()
    return ids, torch.ones_like(ids)


def run_decode(model, decode_inputs, fused: bool):
    from flag_gems.runtime.backend._arm.q4.linear import (
        set_fused_decode_enabled,
    )

    set_fused_decode_enabled(fused)
    begin = time.perf_counter_ns()
    output = model(**decode_inputs)
    token = output.logits[:, -1, :].argmax(dim=-1)
    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
    return elapsed_ms, output.logits[:, -1, :].clone(), token.clone()


def main() -> None:
    args = parse_args()
    if args.pairs <= 0 or args.warmup_pairs < 0:
        raise ValueError("pairs must be positive and warmup-pairs nonnegative")
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    setup = apply_mode("q4-qwen3", model)
    ids, attention_mask = make_prompt(
        tokenizer, args.prompt, args.prompt_tokens
    )

    # Compile both routes and their exact T=1/cache-length specializations.
    with torch.inference_mode():
        for _ in range(args.warmup_pairs):
            for fused in (False, True):
                decode_inputs = _make_decode_inputs(
                    model, ids, attention_mask
                )
                run_decode(model, decode_inputs, fused)

    samples = {"legacy_ms": [], "fused_ms": []}
    paired_ratios = []
    exact = True
    tokens_equal = True
    orders = []
    with torch.inference_mode():
        for pair in range(args.pairs):
            order = (False, True) if pair % 2 == 0 else (True, False)
            orders.append(["fused" if item else "legacy" for item in order])
            results = {}
            for fused in order:
                decode_inputs = _make_decode_inputs(
                    model, ids, attention_mask
                )
                results[fused] = run_decode(model, decode_inputs, fused)
            legacy_ms, legacy_logits, legacy_token = results[False]
            fused_ms, fused_logits, fused_token = results[True]
            samples["legacy_ms"].append(legacy_ms)
            samples["fused_ms"].append(fused_ms)
            paired_ratios.append(fused_ms / legacy_ms)
            exact = exact and torch.equal(fused_logits, legacy_logits)
            tokens_equal = tokens_equal and torch.equal(
                fused_token, legacy_token
            )

    legacy_median = statistics.median(samples["legacy_ms"])
    fused_median = statistics.median(samples["fused_ms"])
    ratio_of_medians = fused_median / legacy_median
    paired_ratio_median = statistics.median(paired_ratios)
    from flag_gems.runtime.backend._arm.q4 import stats as q4_stats
    import triton

    result = {
        "status": "PASS" if exact and tokens_equal else "FAIL",
        "model": args.model,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "prompt_tokens": int(ids.shape[1]),
        "pairs": args.pairs,
        "legacy_decode_median_ms": legacy_median,
        "fused_decode_median_ms": fused_median,
        "ratio_of_medians": ratio_of_medians,
        "paired_ratio_median": paired_ratio_median,
        "paired_improvement_percent": 100.0 * (1.0 - paired_ratio_median),
        "bit_exact_logits": exact,
        "tokens_equal": tokens_equal,
        "samples": samples,
        "paired_ratios": paired_ratios,
        "orders": orders,
        "setup": setup,
        "q4_stats": q4_stats(),
        "torch": torch.__version__,
        "triton": triton.__file__,
        "target": str(triton.runtime.driver.active.get_current_target()),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(encoded, end="")
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    if result["status"] != "PASS":
        raise AssertionError("fused and legacy decode results differ")


if __name__ == "__main__":
    main()
