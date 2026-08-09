#!/usr/bin/env python3
"""Paired real-checkpoint A/B for final RMSNorm plus runtime-Q4 lm_head."""

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

import torch  # noqa: E402

from integrations.minicpm5.load_minicpm5 import load_minicpm5_q4  # noqa: E402


MODEL = Path("/home/cix/MiniCPM5-2.6B-W4A8-GPTQ-G128")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--partitions", type=int, nargs="+")
    parser.add_argument("--json-out")
    return parser.parse_args()


def paired(functions, warmup: int, repeats: int):
    for _ in range(warmup):
        for function in functions.values():
            function()
    samples = {name: [] for name in functions}
    for repeat in range(repeats):
        names = tuple(functions)
        if repeat % 2:
            names = tuple(reversed(names))
        for name in names:
            begin = time.perf_counter_ns()
            functions[name]()
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e3)
    return {
        name: {
            "median_us": statistics.median(values),
            "samples_us": values,
        }
        for name, values in samples.items()
    }


def partitioned(function, partitions: int):
    def call():
        name = "FLAGGEMS_ARM_Q4_DECODE_PARTITIONS"
        previous = os.environ.get(name)
        os.environ[name] = str(partitions)
        try:
            return function()
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    return call


def main():
    args = parse_args()
    if args.threads <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("invalid benchmark iteration count")
    torch.manual_seed(24000)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, _ = load_minicpm5_q4(MODEL, quantize_lm_head=True)
    norm = model.model.norm
    head = model.lm_head
    value = torch.randn((1, 1, head.in_features), dtype=torch.bfloat16)

    with torch.inference_mode():
        staged = lambda: head(norm(value))
        fused = lambda: head.forward_rmsnorm(
            value, norm.weight, norm.variance_epsilon
        )
        staged_output = staged()
        fused_output = fused()
        assert torch.equal(staged_output, fused_output)
        if args.partitions:
            if min(args.partitions) <= 0:
                raise ValueError("partitions must be positive")
            timing = paired(
                {
                    f"staged_partitions_{partitions}": partitioned(
                        staged, partitions
                    )
                    for partitions in args.partitions
                },
                args.warmup,
                args.repeats,
            )
        else:
            timing = paired(
                {
                    "staged_rmsnorm_q4_head": staged,
                    "fused_rmsnorm_pack_once_q4_head": fused,
                },
                args.warmup,
                args.repeats,
            )
            timing["speedup"] = (
                timing["staged_rmsnorm_q4_head"]["median_us"]
                / timing["fused_rmsnorm_pack_once_q4_head"]["median_us"]
            )
    result = {
        "status": "PASS",
        "comparison": "same_runtime_Q4_head_same_output_not_BF16",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "bit_exact": True,
        "timing": timing,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")


if __name__ == "__main__":
    main()
