#!/usr/bin/env python3
"""Real-checkpoint MiniCPM Q4/Q8 microbench: native ATen versus Triton.

The comparison never uses a BF16 ``nn.Linear`` baseline.  Q4 uses PyTorch's
native dynamic-A8/groupwise-Q4 KleidiAI operator.  Q8 reports both the exact
eager-ATen W8A8 graph and the faster native A16W8 control explicitly, since
PyTorch 2.10 does not expose a fused CPU operator for the checkpoint's W8A8
contract.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = Path(
    os.getenv(
        "TRITON_CPU_VENV_SITE",
        "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
    )
)
TRITON_PYTHON = Path(
    os.getenv("TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python")
)
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
    str(VENV_SITE),
]
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (  # noqa: E402
    TLEInt8Linear,
)
from flag_gems.runtime.backend._arm.q4.aten_linear import (  # noqa: E402
    AtenQ4Linear,
)
from flag_gems.runtime.backend._arm.q4.optimize_qwen3 import (  # noqa: E402
    Q4Linear,
)
from integrations.minicpm5.load_minicpm5 import (  # noqa: E402
    _AtenW8A8Linear,
    _AtenW8WeightOnlyLinear,
)


Q4_MODEL = Path("/home/cix/MiniCPM5-2.6B-W4A8-GPTQ-G128")
W8_MODEL = Path("/home/cix/MiniCPM5-2.6B-W8A8-CT")
LAYER_PREFIX = "model.layers.0"
LINEARS = {
    "q_proj": f"{LAYER_PREFIX}.self_attn.q_proj",
    "k_proj": f"{LAYER_PREFIX}.self_attn.k_proj",
    "v_proj": f"{LAYER_PREFIX}.self_attn.v_proj",
    "o_proj": f"{LAYER_PREFIX}.self_attn.o_proj",
    "gate_proj": f"{LAYER_PREFIX}.mlp.gate_proj",
    "up_proj": f"{LAYER_PREFIX}.mlp.up_proj",
    "down_proj": f"{LAYER_PREFIX}.mlp.down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=("q4", "w8", "both"), default="both")
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 4, 12, 16))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--q4-blocks",
        type=int,
        nargs="+",
        choices=(4, 8, 12, 16),
        help="A/B G128 prefill BLOCK_M values in the same process",
    )
    parser.add_argument(
        "--q4-decode-partitions",
        type=int,
        nargs="+",
        help="A/B G128 decode output-partition counts in the same process",
    )
    parser.add_argument("--json-out")
    return parser.parse_args()


def paired_bench(functions, value, warmup: int, repeats: int):
    names = tuple(functions)
    for _ in range(warmup):
        for name in names:
            functions[name](value)
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            begin = time.perf_counter_ns()
            functions[name](value)
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e3)
    return {
        name: {
            "median_us": statistics.median(values),
            "samples_us": values,
        }
        for name, values in samples.items()
    }


def error(candidate: torch.Tensor, reference: torch.Tensor):
    difference = candidate.float() - reference.float()
    denominator = torch.linalg.vector_norm(reference.float())
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / denominator
        ),
        "finite": bool(torch.isfinite(candidate).all()),
    }


def q4_environment_function(linear, values):
    """Bind production-router overrides without leaking global state."""

    def call(value):
        previous = {name: os.environ.get(name) for name in values}
        os.environ.update({name: str(setting) for name, setting in values.items()})
        try:
            return linear(value)
        finally:
            for name, setting in previous.items():
                if setting is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = setting

    return call


def q4_timing_functions(args, aten_function, triton_linear):
    functions = {"aten_kleidiai": aten_function}
    if args.q4_blocks:
        functions.update(
            {
                f"triton_block_m{block_m}": q4_environment_function(
                    triton_linear,
                    {"FLAGGEMS_ARM_Q4_G128_PREFILL_BLOCK_M": block_m},
                )
                for block_m in args.q4_blocks
            }
        )
    elif args.q4_decode_partitions:
        functions.update(
            {
                f"triton_partitions_{partitions}": q4_environment_function(
                    triton_linear,
                    {"FLAGGEMS_ARM_Q4_DECODE_PARTITIONS": partitions},
                )
                for partitions in args.q4_decode_partitions
            }
        )
    else:
        functions["triton"] = triton_linear
    return functions


def q4_linear(checkpoint, key: str):
    weight = checkpoint.get_tensor(f"{key}.weight")
    scale = checkpoint.get_tensor(f"{key}.weight_scale")
    return (
        Q4Linear.from_grouped_int4(weight, scale, group_size=128),
        AtenQ4Linear.from_grouped_int4(weight, scale, group_size=128),
        tuple(weight.shape),
    )


def bench_q4(args):
    results = {}
    checkpoint_path = Q4_MODEL / "model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for name in ("q_proj", "o_proj", "gate_proj", "down_proj"):
            triton_linear, aten_linear, (n, k) = q4_linear(
                checkpoint, LINEARS[name]
            )
            cases = {}
            for rows in args.rows:
                torch.manual_seed(1000 + rows)
                value = torch.randn((rows, k), dtype=torch.bfloat16)
                aten_output = aten_linear(value)
                triton_output = triton_linear(value)
                timing = paired_bench(
                    q4_timing_functions(args, aten_linear, triton_linear),
                    value,
                    args.warmup,
                    args.repeats,
                )
                ratios = {
                    name: result["median_us"]
                    / timing["aten_kleidiai"]["median_us"]
                    for name, result in timing.items()
                    if name.startswith("triton")
                }
                timing["triton_over_aten"] = (
                    ratios
                    if args.q4_blocks or args.q4_decode_partitions
                    else ratios["triton"]
                )
                cases[str(rows)] = {
                    "timing": timing,
                    "triton_vs_aten": error(triton_output, aten_output),
                }
            results[name] = {"shape_nk": [n, k], "rows": cases}

        # Model-level joined matrices are reported independently so fusion
        # cannot disguise a regression in the standalone linear result.
        for joined_name, members in {
            "joined_qkv": ("q_proj", "k_proj", "v_proj"),
            "joined_gate_up": ("gate_proj", "up_proj"),
        }.items():
            weights = [
                checkpoint.get_tensor(f"{LINEARS[name]}.weight")
                for name in members
            ]
            scales = [
                checkpoint.get_tensor(f"{LINEARS[name]}.weight_scale")
                for name in members
            ]
            joined_weight = torch.cat(weights, dim=0)
            joined_scale = torch.cat(scales, dim=0)
            n, k = joined_weight.shape
            triton_linear = Q4Linear.from_grouped_int4(
                joined_weight, joined_scale, group_size=128
            )
            aten_linears = [q4_linear(checkpoint, LINEARS[name])[1] for name in members]

            def aten_joined(value):
                return torch.cat([linear(value) for linear in aten_linears], dim=-1)

            cases = {}
            for rows in args.rows:
                torch.manual_seed(2000 + rows)
                value = torch.randn((rows, k), dtype=torch.bfloat16)
                aten_output = aten_joined(value)
                triton_output = triton_linear(value)
                functions = q4_timing_functions(
                    args, aten_joined, triton_linear
                )
                functions["aten_independent"] = functions.pop(
                    "aten_kleidiai"
                )
                if "triton" in functions:
                    functions["triton_joined"] = functions.pop("triton")
                timing = paired_bench(
                    functions, value, args.warmup, args.repeats
                )
                ratios = {
                    name: result["median_us"]
                    / timing["aten_independent"]["median_us"]
                    for name, result in timing.items()
                    if name.startswith("triton")
                }
                timing["triton_over_aten"] = (
                    ratios
                    if args.q4_blocks or args.q4_decode_partitions
                    else ratios["triton_joined"]
                )
                cases[str(rows)] = {
                    "timing": timing,
                    "triton_vs_aten": error(triton_output, aten_output),
                }
            results[joined_name] = {
                "members": list(members),
                "shape_nk": [n, k],
                "rows": cases,
            }
    return results


def bench_w8(args):
    results = {}
    checkpoint_path = W8_MODEL / "model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for name in ("q_proj", "o_proj", "gate_proj", "down_proj"):
            key = LINEARS[name]
            weight = checkpoint.get_tensor(f"{key}.weight")
            scale = checkpoint.get_tensor(f"{key}.weight_scale")
            n, k = weight.shape
            triton_linear = TLEInt8Linear(weight, scale)
            aten_exact = _AtenW8A8Linear(weight, scale)
            aten_weight_only = _AtenW8WeightOnlyLinear(weight, scale)
            cases = {}
            for rows in args.rows:
                torch.manual_seed(3000 + rows)
                value = torch.randn((rows, k), dtype=torch.bfloat16)
                exact_output = aten_exact(value)
                triton_output = triton_linear(value)
                weight_only_output = aten_weight_only(value)
                timing = paired_bench(
                    {
                        "aten_exact_w8a8": aten_exact,
                        "aten_a16w8_control": aten_weight_only,
                        "triton_w8a8": triton_linear,
                    },
                    value,
                    args.warmup,
                    args.repeats,
                )
                timing["triton_over_aten_exact"] = (
                    timing["triton_w8a8"]["median_us"]
                    / timing["aten_exact_w8a8"]["median_us"]
                )
                timing["triton_over_aten_a16w8"] = (
                    timing["triton_w8a8"]["median_us"]
                    / timing["aten_a16w8_control"]["median_us"]
                )
                cases[str(rows)] = {
                    "timing": timing,
                    "triton_vs_exact": error(triton_output, exact_output),
                    "a16w8_vs_exact": error(weight_only_output, exact_output),
                }
            results[name] = {"shape_nk": [n, k], "rows": cases}
    return results


def main() -> None:
    args = parse_args()
    if min(args.rows) <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("rows/repeats must be positive and warmup nonnegative")
    if args.q4_blocks and args.q4_decode_partitions:
        raise ValueError("select only one Q4 router A/B dimension")
    if args.q4_decode_partitions and min(args.q4_decode_partitions) <= 0:
        raise ValueError("Q4 decode partitions must be positive")
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with torch.inference_mode():
        result = {
            "status": "PASS",
            "comparison": "quantized_checkpoint_vs_native_aten_not_bf16",
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "threads": args.threads,
            "rows": args.rows,
            "torch": torch.__version__,
            "kleidiai": bool(torch.backends.kleidiai.is_available()),
        }
        if args.quant in ("q4", "both"):
            result["q4"] = bench_q4(args)
        if args.quant in ("w8", "both"):
            result["w8"] = bench_w8(args)
    import triton

    result["triton"] = triton.__file__
    result["target"] = str(triton.runtime.driver.active.get_current_target())
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(encoded, end="")
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)


if __name__ == "__main__":
    main()
