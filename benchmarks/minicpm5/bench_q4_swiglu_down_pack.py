#!/usr/bin/env python3
"""Paired real-checkpoint A/B for Q4 SwiGLU-to-down compact packing."""

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

from flag_gems.runtime.backend._arm.q4.optimize_qwen3 import (  # noqa: E402
    set_fused_down_residual_enabled,
    set_fused_swiglu_down_pack_enabled,
)
from integrations.minicpm5.load_minicpm5 import (  # noqa: E402
    load_minicpm5_q4,
)


MODEL = Path("/home/cix/MiniCPM5-2.6B-W4A8-GPTQ-G128")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--json-out")
    return parser.parse_args()


def route(function, enabled: bool):
    previous = set_fused_swiglu_down_pack_enabled(enabled)
    try:
        return function()
    finally:
        set_fused_swiglu_down_pack_enabled(previous)


def residual_route(function, enabled: bool):
    previous = set_fused_down_residual_enabled(enabled)
    try:
        return function()
    finally:
        set_fused_down_residual_enabled(previous)


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


def main():
    args = parse_args()
    if args.threads <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("invalid benchmark iteration count")
    torch.manual_seed(24000)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, _ = load_minicpm5_q4(MODEL)
    mlp = model.model.layers[0].mlp
    norm = model.model.layers[0].post_attention_layernorm
    value = torch.randn((1, 1, 2048), dtype=torch.bfloat16)
    residual = torch.randn_like(value)
    with torch.inference_mode():
        joined = mlp.gate_up(value)
        legacy_finish = lambda: route(
            lambda: mlp._finish(joined, value.shape), False
        )
        fused_finish = lambda: route(
            lambda: mlp._finish(joined, value.shape), True
        )
        legacy_output = legacy_finish()
        fused_output = fused_finish()
        assert torch.equal(legacy_output, fused_output)
        finish = paired(
            {"legacy_swiglu_then_down": legacy_finish,
             "fused_swiglu_compact_pack_down": fused_finish},
            args.warmup,
            args.repeats,
        )

        legacy_mlp = lambda: route(lambda: mlp(value), False)
        fused_mlp = lambda: route(lambda: mlp(value), True)
        assert torch.equal(legacy_mlp(), fused_mlp())
        whole_mlp = paired(
            {"legacy_mlp": legacy_mlp, "fused_mlp": fused_mlp},
            args.warmup,
            args.repeats,
        )

        staged_residual = lambda: residual_route(
            lambda: mlp.forward_add_rmsnorm_and_residual(
                value, residual, norm
            ),
            False,
        )
        fused_residual = lambda: residual_route(
            lambda: mlp.forward_add_rmsnorm_and_residual(
                value, residual, norm
            ),
            True,
        )
        assert torch.equal(staged_residual(), fused_residual())
        full_tail = paired(
            {
                "staged_final_residual": staged_residual,
                "down_store_residual_epilogue": fused_residual,
            },
            args.warmup,
            args.repeats,
        )


    finish["speedup"] = (
        finish["legacy_swiglu_then_down"]["median_us"]
        / finish["fused_swiglu_compact_pack_down"]["median_us"]
    )
    whole_mlp["speedup"] = (
        whole_mlp["legacy_mlp"]["median_us"]
        / whole_mlp["fused_mlp"]["median_us"]
    )
    full_tail["speedup"] = (
        full_tail["staged_final_residual"]["median_us"]
        / full_tail["down_store_residual_epilogue"]["median_us"]
    )
    result = {
        "status": "PASS",
        "comparison": "same_Q4_checkpoint_same_output_not_BF16",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "bit_exact": True,
        "finish": finish,
        "whole_mlp": whole_mlp,
        "post_attention_norm_mlp_residual": full_tail,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")


if __name__ == "__main__":
    main()
