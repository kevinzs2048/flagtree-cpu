#!/usr/bin/env python3
"""Minimal end-to-end Qwen3.5 W4A8 smoke test for vLLM on Apple Silicon."""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from flag_gems.runtime.backend._arm.q4 import linear as q4_linear
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        language_model_only=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    print(f"MODEL_LOAD_SECONDS={time.perf_counter() - started:.3f}", flush=True)
    print(f"ROUTER_STATS_AFTER_LOAD={q4_linear._STATS}", flush=True)

    started = time.perf_counter()
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0, max_tokens=args.max_tokens),
    )
    generated = outputs[0].outputs[0]
    print(f"GENERATE_SECONDS={time.perf_counter() - started:.3f}", flush=True)
    print(f"GENERATED_TOKEN_IDS={generated.token_ids}", flush=True)
    print(f"GENERATED_TEXT={generated.text!r}", flush=True)
    print(f"ROUTER_STATS_AFTER_GENERATE={q4_linear._STATS}", flush=True)


if __name__ == "__main__":
    main()
