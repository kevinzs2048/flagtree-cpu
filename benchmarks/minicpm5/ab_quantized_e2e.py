#!/usr/bin/env python3
"""Same-process paired MiniCPM quantized end-to-end A/B benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from run_minicpm5 import (  # establishes the pinned local source paths
    MODEL_DIRS,
    AutoTokenizer,
    load,
    make_decode_inputs,
    make_prompt,
    torch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, choices=tuple(MODEL_DIRS))
    parser.add_argument("--right", required=True, choices=tuple(MODEL_DIRS))
    parser.add_argument("--prompt", default="What is 2+3? Think step by step.")
    parser.add_argument("--prompt-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--quantize-lm-head", action="store_true")
    parser.add_argument(
        "--q4-down-residual-ab",
        action="store_true",
        help="allow q4/q4 and enable the down-store epilogue only on left",
    )
    parser.add_argument("--json-out")
    return parser.parse_args()


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def load_mode(mode: str, quantize_lm_head: bool):
    args = SimpleNamespace(mode=mode, quantize_lm_head=quantize_lm_head)
    return load(args, MODEL_DIRS[mode])


def elapsed_ms(function):
    begin = time.perf_counter_ns()
    result = function()
    return (time.perf_counter_ns() - begin) / 1.0e6, result


def main() -> None:
    args = parse_args()
    if args.left == args.right and not args.q4_down_residual_ab:
        raise ValueError("left and right modes must differ")
    if args.q4_down_residual_ab and (args.left, args.right) != ("q4", "q4"):
        raise ValueError("down-residual A/B requires --left q4 --right q4")
    if args.q4_down_residual_ab:
        from flag_gems.runtime.backend._arm.q4.optimize_qwen3 import (
            set_fused_down_residual_enabled,
        )

        def invoke(name, function):
            previous = set_fused_down_residual_enabled(name == "left")
            try:
                return function()
            finally:
                set_fused_down_residual_enabled(previous)
    else:
        def invoke(name, function):
            return function()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIRS[args.left], local_files_only=True
    )
    models = {}
    setups = {}
    load_seconds = {}
    for name, mode in (("left", args.left), ("right", args.right)):
        models[name], setups[name], load_seconds[name] = load_mode(
            mode, args.quantize_lm_head
        )
    ids, attention_mask = make_prompt(
        tokenizer, args.prompt, args.prompt_tokens
    )

    with torch.inference_mode():
        for _ in range(args.warmup_pairs):
            for name in ("left", "right"):
                decode_inputs = make_decode_inputs(
                    models[name], ids, attention_mask
                )
                invoke(name, lambda: models[name](**decode_inputs))

        samples = {
            "left_prefill_ms": [],
            "right_prefill_ms": [],
            "left_decode_ms": [],
            "right_decode_ms": [],
        }
        paired_prefill_ratios = []
        paired_decode_ratios = []
        last_logits = {}
        last_tokens = {}
        orders = []
        for pair in range(args.pairs):
            order = (
                ("left", "right")
                if pair % 2 == 0
                else ("right", "left")
            )
            orders.append(list(order))
            prefill_elapsed = {}
            decode_elapsed = {}
            for name in order:
                prefill_elapsed[name], _ = elapsed_ms(
                    lambda name=name: invoke(
                        name,
                        lambda: models[name](
                            input_ids=ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                        ),
                    )
                )
                decode_inputs = make_decode_inputs(
                    models[name], ids, attention_mask
                )
                decode_elapsed[name], output = elapsed_ms(
                    lambda name=name, decode_inputs=decode_inputs: invoke(
                        name,
                        lambda: models[name](**decode_inputs),
                    )
                )
                last_logits[name] = output.logits[:, -1].clone()
                last_tokens[name] = int(
                    last_logits[name].argmax(dim=-1).item()
                )
            for name in ("left", "right"):
                samples[f"{name}_prefill_ms"].append(prefill_elapsed[name])
                samples[f"{name}_decode_ms"].append(decode_elapsed[name])
            paired_prefill_ratios.append(
                prefill_elapsed["left"] / prefill_elapsed["right"]
            )
            paired_decode_ratios.append(
                decode_elapsed["left"] / decode_elapsed["right"]
            )

    difference = last_logits["left"].float() - last_logits["right"].float()
    result = {
        "status": "PASS",
        "left": args.left,
        "right": args.right,
        "comparison": "quantized_checkpoint_vs_quantized_checkpoint_not_bf16",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "prompt_tokens": int(ids.shape[1]),
        "pairs": args.pairs,
        "quantize_lm_head": args.quantize_lm_head,
        "q4_down_residual_ab": args.q4_down_residual_ab,
        "load_seconds": load_seconds,
        "rss_mib": rss_mib(),
        "samples": samples,
        "orders": orders,
        "left_prefill_median_ms": statistics.median(
            samples["left_prefill_ms"]
        ),
        "right_prefill_median_ms": statistics.median(
            samples["right_prefill_ms"]
        ),
        "left_decode_median_ms": statistics.median(
            samples["left_decode_ms"]
        ),
        "right_decode_median_ms": statistics.median(
            samples["right_decode_ms"]
        ),
        "paired_prefill_ratio_median": statistics.median(
            paired_prefill_ratios
        ),
        "paired_decode_ratio_median": statistics.median(
            paired_decode_ratios
        ),
        "tokens": last_tokens,
        "logits": {
            "max_abs": float(difference.abs().max()),
            "mean_abs": float(difference.abs().mean()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(last_logits["right"].float())
            ),
            "finite": {
                name: bool(torch.isfinite(value).all())
                for name, value in last_logits.items()
            },
        },
        "setups": setups,
        "torch": torch.__version__,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    print(encoded, end="")
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)


if __name__ == "__main__":
    main()
