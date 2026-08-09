#!/usr/bin/env python3
"""Same-process Qwen Q4 decode A/B: fused Triton versus ATen/KleidiAI."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from run_qwen3 import (  # establishes explicit local source import paths
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
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--json-out")
    return parser.parse_args()


def load_model(path: str):
    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()


def make_prompt(tokenizer, text: str, prompt_tokens: int):
    ids = tokenizer(text, return_tensors="pt").input_ids
    if prompt_tokens:
        repeats = (prompt_tokens + ids.shape[1] - 1) // ids.shape[1]
        ids = ids.repeat(1, repeats)[:, :prompt_tokens].contiguous()
    return ids, torch.ones_like(ids)


def timed_decode(model, inputs):
    begin = time.perf_counter_ns()
    output = model(**inputs)
    token = output.logits[:, -1, :].argmax(dim=-1)
    return (time.perf_counter_ns() - begin) / 1e6, token.item()


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def main() -> None:
    args = parse_args()
    if args.pairs <= 0 or args.warmup_pairs < 0:
        raise ValueError("pairs must be positive and warmup-pairs nonnegative")
    if not torch.backends.kleidiai.is_available():
        raise RuntimeError("ATen/KleidiAI is unavailable in this PyTorch build")
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    triton_model = load_model(args.model)
    triton_setup = apply_mode("q4-qwen3", triton_model)
    aten_model = load_model(args.model)
    aten_setup = apply_mode("q4-aten", aten_model)
    ids, attention_mask = make_prompt(
        tokenizer, args.prompt, args.prompt_tokens
    )
    models = {"triton": triton_model, "aten": aten_model}

    with torch.inference_mode():
        for _ in range(args.warmup_pairs):
            for name in ("triton", "aten"):
                inputs = _make_decode_inputs(
                    models[name], ids, attention_mask
                )
                timed_decode(models[name], inputs)

    samples = {"triton_ms": [], "aten_ms": []}
    ratios = []
    tokens = {"triton": [], "aten": []}
    orders = []
    with torch.inference_mode():
        for pair in range(args.pairs):
            order = ("triton", "aten") if pair % 2 == 0 else (
                "aten",
                "triton",
            )
            orders.append(list(order))
            elapsed = {}
            for name in order:
                inputs = _make_decode_inputs(
                    models[name], ids, attention_mask
                )
                elapsed[name], token = timed_decode(models[name], inputs)
                tokens[name].append(token)
            samples["triton_ms"].append(elapsed["triton"])
            samples["aten_ms"].append(elapsed["aten"])
            ratios.append(elapsed["triton"] / elapsed["aten"])

    triton_median = statistics.median(samples["triton_ms"])
    aten_median = statistics.median(samples["aten_ms"])
    ratio_median = statistics.median(ratios)
    import triton

    result = {
        "status": "PASS",
        "model": args.model,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "prompt_tokens": int(ids.shape[1]),
        "pairs": args.pairs,
        "triton_decode_median_ms": triton_median,
        "aten_decode_median_ms": aten_median,
        "ratio_of_medians": triton_median / aten_median,
        "paired_ratio_median": ratio_median,
        "triton_delta_percent": 100.0 * (ratio_median - 1.0),
        "samples": samples,
        "paired_ratios": ratios,
        "orders": orders,
        "tokens": tokens,
        "triton_setup": triton_setup,
        "aten_setup": aten_setup,
        "rss_mib": rss_mib(),
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


if __name__ == "__main__":
    main()
