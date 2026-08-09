#!/usr/bin/env python3
"""PyTorch end-to-end runner for local MiniCPM5 BF16/Q4/Q8 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "third_party/FlagGems/src"),
    str(ROOT / "ports/triton-cpu-3.7.2/python"),
    "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")
os.environ.setdefault("FLAGGEMS_ARM_ATTN_DISABLE_RUNTIME", "1")
os.environ.setdefault("FLAGGEMS_ARM_ATTN_SHORT_PREFILL_CODEGEN", "1")
if "--profile" in sys.argv:
    os.environ["FLAGGEMS_PROFILE_RANGES"] = "1"

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from integrations.minicpm5 import (  # noqa: E402
    load_minicpm5_q4,
    load_minicpm5_q4_aten,
    load_minicpm5_q4_aten_fused,
    load_minicpm5_q4_reference,
    load_minicpm5_w8,
    load_minicpm5_w8_aten,
)

MODEL_DIRS = {
    "bf16": Path(
        "/home/cix/MiniCPM5-2.6B-0426_job_327123_step_24000_fusion_think_512k"
    ),
    "q4": Path("/home/cix/MiniCPM5-2.6B-W4A8-GPTQ-G128"),
    "q4_aten": Path("/home/cix/MiniCPM5-2.6B-W4A8-GPTQ-G128"),
    "q4_aten_fused": Path("/home/cix/MiniCPM5-2.6B-W4A8-GPTQ-G128"),
    "q4_ref": Path("/home/cix/MiniCPM5-2.6B-W4A8-GPTQ-G128"),
    "w8": Path("/home/cix/MiniCPM5-2.6B-W8A8-CT"),
    "w8_aten": Path("/home/cix/MiniCPM5-2.6B-W8A8-CT"),
    "w8_aten_weight_only": Path("/home/cix/MiniCPM5-2.6B-W8A8-CT"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODEL_DIRS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="What is 2+3? Think step by step.")
    parser.add_argument("--prompt-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--decode-repeats", type=int, default=7)
    parser.add_argument("--new-tokens", type=int, default=2)
    lm_head = parser.add_mutually_exclusive_group()
    lm_head.add_argument(
        "--quantize-lm-head",
        action="store_true",
        help=(
            "also runtime-quantize the checkpoint's ignored BF16 lm_head; "
            "faster, but not checkpoint-fidelity mode"
        ),
    )
    # Keep old local commands working while making checkpoint fidelity the
    # safe default for new invocations.
    lm_head.add_argument(
        "--no-quant-lm-head",
        action="store_false",
        dest="quantize_lm_head",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(quantize_lm_head=False)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--json-out")
    parser.add_argument(
        "--logits-out",
        help="optional torch.save path for the final timed decode logits",
    )
    return parser.parse_args()


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def load(args, model_dir: Path):
    started = time.perf_counter()
    if args.mode == "q4":
        model, setup = load_minicpm5_q4(
            model_dir, quantize_lm_head=args.quantize_lm_head
        )
    elif args.mode == "q4_aten":
        model, setup = load_minicpm5_q4_aten(
            model_dir, quantize_lm_head=args.quantize_lm_head
        )
    elif args.mode == "q4_aten_fused":
        model, setup = load_minicpm5_q4_aten_fused(
            model_dir, quantize_lm_head=args.quantize_lm_head
        )
    elif args.mode == "q4_ref":
        model, setup = load_minicpm5_q4_reference(model_dir)
    elif args.mode == "w8":
        model, setup = load_minicpm5_w8(
            model_dir, quantize_lm_head=args.quantize_lm_head
        )
    elif args.mode in ("w8_aten", "w8_aten_weight_only"):
        model, setup = load_minicpm5_w8_aten(
            model_dir,
            weight_only=args.mode == "w8_aten_weight_only",
            quantize_lm_head=args.quantize_lm_head,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
        setup = {"quantization": "bf16_reference"}
    return model, setup, time.perf_counter() - started


def make_prompt(tokenizer, text: str, prompt_tokens: int):
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded.input_ids
    if prompt_tokens:
        repeats = (prompt_tokens + ids.shape[1] - 1) // ids.shape[1]
        ids = ids.repeat(1, repeats)[:, :prompt_tokens].contiguous()
    return ids, torch.ones_like(ids)


def make_decode_inputs(model, ids, attention_mask):
    prefill = model(
        input_ids=ids, attention_mask=attention_mask, use_cache=True
    )
    next_id = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    return {
        "input_ids": next_id,
        "attention_mask": torch.ones(
            (1, ids.shape[1] + 1), dtype=attention_mask.dtype
        ),
        "past_key_values": prefill.past_key_values,
        "cache_position": torch.tensor([ids.shape[1]], dtype=torch.long),
        "use_cache": True,
    }


def profile_decode(model, decode_inputs):
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        output = model(**decode_inputs)
        output.logits[:, -1].argmax(dim=-1)
    events = list(prof.key_averages())
    total = sum(event.self_cpu_time_total for event in events)
    triton_events = [event for event in events if event.key.startswith("triton::")]
    triton_total = sum(event.self_cpu_time_total for event in triton_events)
    return {
        "total_self_cpu_us": total,
        "triton_self_cpu_us": triton_total,
        "triton_coverage": triton_total / total if total else 0.0,
        "triton_ranges": {
            event.key: {"calls": event.count, "self_cpu_us": event.self_cpu_time_total}
            for event in sorted(
                triton_events,
                key=lambda item: item.self_cpu_time_total,
                reverse=True,
            )
        },
        "top_non_triton": [
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_us": event.self_cpu_time_total,
            }
            for event in sorted(
                (item for item in events if not item.key.startswith("triton::")),
                key=lambda item: item.self_cpu_time_total,
                reverse=True,
            )[:20]
        ],
    }


def main() -> None:
    args = parse_args()
    if args.profile:
        os.environ["FLAGGEMS_PROFILE_RANGES"] = "1"
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model_dir = Path(args.model) if args.model else MODEL_DIRS[args.mode]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model, setup, load_s = load(args, model_dir)
    ids, attention_mask = make_prompt(
        tokenizer, args.prompt, args.prompt_tokens
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            inputs = make_decode_inputs(model, ids, attention_mask)
            model(**inputs)

        prefill_ms = []
        prefill_output = None
        for _ in range(args.prefill_repeats):
            begin = time.perf_counter_ns()
            prefill_output = model(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            prefill_ms.append((time.perf_counter_ns() - begin) / 1e6)

        decode_ms = []
        logits = None
        token = None
        for _ in range(args.decode_repeats):
            inputs = make_decode_inputs(model, ids, attention_mask)
            begin = time.perf_counter_ns()
            output = model(**inputs)
            logits = output.logits[:, -1].clone()
            token = logits.argmax(dim=-1)
            decode_ms.append((time.perf_counter_ns() - begin) / 1e6)

        generated = model.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=args.new_tokens,
            min_new_tokens=args.new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        profile_summary = None
        if args.profile:
            profile_summary = profile_decode(
                model, make_decode_inputs(model, ids, attention_mask)
            )

    assert prefill_output is not None and logits is not None and token is not None
    result = {
        "status": "PASS",
        "mode": args.mode,
        "model": str(model_dir),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "prompt_tokens": int(ids.shape[1]),
        "load_s": load_s,
        "rss_mib": rss_mib(),
        "setup": setup,
        "prefill_ms": prefill_ms,
        "prefill_median_ms": statistics.median(prefill_ms),
        "decode_ms": decode_ms,
        "decode_median_ms": statistics.median(decode_ms),
        "decode_token": int(token.item()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_sum": float(logits.float().sum()),
        "generated_token_ids": generated[0, ids.shape[1] :].tolist(),
        "generated_text": tokenizer.decode(
            generated[0, ids.shape[1] :], skip_special_tokens=False
        ),
        "torch": torch.__version__,
    }
    if profile_summary is not None:
        result["profile"] = profile_summary
    try:
        import triton

        result["triton"] = triton.__file__
        result["target"] = str(triton.runtime.driver.active.get_current_target())
    except ImportError:
        result["triton"] = None
    encoded = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    print(encoded, end="")
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    if args.logits_out:
        path = Path(args.logits_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(logits.cpu(), path)


if __name__ == "__main__":
    main()
