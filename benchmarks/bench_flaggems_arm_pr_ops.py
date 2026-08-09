#!/usr/bin/env python3
"""Microbenchmark the curated Arm operators added by FlagGems PR 3616/3775.

Run this script in a fresh process for each FlagGems/compiler checkout.  It
calls the backend functions directly, so process-wide aten overrides cannot
contaminate the reference measurement.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_WORKTREE_FLAGGEMS = (
    Path(__file__).resolve().parents[1] / "third_party" / "FlagGems" / "src"
)
sys.path.insert(0, str(_WORKTREE_FLAGGEMS))

import torch

# The end-to-end venv contains a scikit-build editable finder for a separate
# FlagGems checkout.  Respect the checkout selected through PYTHONPATH rather
# than allowing that finder to intercept imports before PathFinder.
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ != "_flag_gems_editable"
]


@dataclass
class Case:
    name: str
    aten: Callable[[], object]
    gems: Callable[[], object]
    check: Callable[[object, object], None]


def tensor_check(atol=0.0, rtol=0.0):
    def check(actual, expected):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    return check


def tuple_check(atol=0.0, rtol=0.0):
    def check(actual, expected):
        if len(actual) != len(expected):
            raise AssertionError("tuple output length differs")
        for lhs, rhs in zip(actual, expected):
            torch.testing.assert_close(lhs, rhs, atol=atol, rtol=rtol)

    return check


def load_op(module: str, function: str):
    full_name = f"flag_gems.runtime.backend._arm.ops.{module}"
    return getattr(importlib.import_module(full_name), function)


def make_cases(
    element_size: int = 131072, vocab_size: int = 151936
) -> dict[str, Case]:
    torch.manual_seed(3616)
    cases: dict[str, Case] = {}

    x_vocab_bf16 = torch.randn(1, vocab_size, dtype=torch.bfloat16)
    fg_argmax = load_op("argmax", "argmax")
    cases["argmax_vocab_bf16"] = Case(
        "argmax_vocab_bf16",
        lambda: torch.argmax(x_vocab_bf16, dim=-1),
        lambda: fg_argmax(x_vocab_bf16, dim=-1),
        tensor_check(),
    )

    x_vocab_f32 = torch.rand(1, vocab_size, dtype=torch.float32)
    fg_topk = load_op("topk", "topk")
    cases["topk_vocab_k50_f32"] = Case(
        "topk_vocab_k50_f32",
        lambda: torch.topk(x_vocab_f32, 50, dim=-1),
        lambda: fg_topk(x_vocab_f32, 50, dim=-1),
        tuple_check(),
    )

    fg_cumsum = load_op("cumsum", "cumsum")
    cases["cumsum_vocab_f32"] = Case(
        "cumsum_vocab_f32",
        lambda: torch.cumsum(x_vocab_f32, dim=-1),
        lambda: fg_cumsum(x_vocab_f32, dim=-1),
        tensor_check(atol=2.0e-2, rtol=2.0e-6),
    )

    fg_min = load_op("min", "min")
    cases["min_vocab_bf16"] = Case(
        "min_vocab_bf16",
        lambda: torch.min(x_vocab_bf16),
        lambda: fg_min(x_vocab_bf16),
        tensor_check(),
    )

    x_element = torch.randn(element_size, dtype=torch.bfloat16)
    y_element = torch.randn(element_size, dtype=torch.bfloat16)
    mask = torch.rand(element_size) > 0.5
    fg_masked_fill = load_op("masked_fill", "masked_fill")
    cases["masked_fill_bf16"] = Case(
        "masked_fill_bf16",
        lambda: torch.masked_fill(x_element, mask, -float("inf")),
        lambda: fg_masked_fill(x_element, mask, -float("inf")),
        tensor_check(),
    )

    fg_where = load_op("where", "where_self")
    cases["where_bf16"] = Case(
        "where_bf16",
        lambda: torch.where(mask, x_element, y_element),
        lambda: fg_where(mask, x_element, y_element),
        tensor_check(),
    )
    fg_where_scalar = load_op("where", "where_scalar_self")
    cases["where_scalar_bf16"] = Case(
        "where_scalar_bf16",
        lambda: torch.where(mask, 0.25, y_element),
        lambda: fg_where_scalar(mask, 0.25, y_element),
        tensor_check(),
    )

    fg_sub = load_op("sub", "sub")
    cases["sub_bf16"] = Case(
        "sub_bf16",
        lambda: torch.sub(x_element, y_element),
        lambda: fg_sub(x_element, y_element),
        tensor_check(),
    )

    fg_div = load_op("div", "true_divide")
    cases["div_scalar_bf16"] = Case(
        "div_scalar_bf16",
        lambda: torch.true_divide(x_element, 1.7),
        lambda: fg_div(x_element, 1.7),
        tensor_check(),
    )

    fg_pow = load_op("pow", "pow_tensor_scalar")
    cases["pow_square_bf16"] = Case(
        "pow_square_bf16",
        lambda: torch.pow(x_element, 2.0),
        lambda: fg_pow(x_element, 2.0),
        tensor_check(),
    )

    fg_lt = load_op("lt", "lt")
    cases["lt_bf16"] = Case(
        "lt_bf16",
        lambda: torch.lt(x_element, y_element),
        lambda: fg_lt(x_element, y_element),
        tensor_check(),
    )

    bool_vector = torch.rand(element_size) > 0.001
    fg_all = load_op("all", "all")
    cases["all_bool"] = Case(
        "all_bool",
        lambda: torch.all(bool_vector),
        lambda: fg_all(bool_vector),
        tensor_check(),
    )
    fg_any = load_op("any", "any")
    cases["any_bool"] = Case(
        "any_bool",
        lambda: torch.any(bool_vector),
        lambda: fg_any(bool_vector),
        tensor_check(),
    )

    fg_full = load_op("full", "full")
    cases["full_bf16"] = Case(
        "full_bf16",
        lambda: torch.full((element_size,), 0.125, dtype=torch.bfloat16),
        lambda: fg_full((element_size,), 0.125, dtype=torch.bfloat16),
        tensor_check(),
    )

    gate = torch.randn(6912, dtype=torch.bfloat16)
    up = torch.randn(6912, dtype=torch.bfloat16)
    fg_swiglu = load_op("silu_and_mul", "arm_silu_and_mul")
    cases["swiglu_decode_bf16"] = Case(
        "swiglu_decode_bf16",
        lambda: torch.nn.functional.silu(gate) * up,
        lambda: fg_swiglu(gate, up),
        tensor_check(),
    )

    table = torch.randn(32768, 256, dtype=torch.bfloat16)
    indices = torch.randint(0, table.shape[0], (64,), dtype=torch.int64)
    fg_index_select = load_op("index_select", "index_select")
    cases["index_select_bf16"] = Case(
        "index_select_bf16",
        lambda: torch.index_select(table, 0, indices),
        lambda: fg_index_select(table, 0, indices),
        tensor_check(),
    )

    gather_input = torch.randn(128, 1024, dtype=torch.bfloat16)
    gather_index = torch.randint(0, 1024, (128, 1024), dtype=torch.int64)
    fg_gather = load_op("gather", "gather")
    cases["gather_bf16"] = Case(
        "gather_bf16",
        lambda: torch.gather(gather_input, 1, gather_index),
        lambda: fg_gather(gather_input, 1, gather_index),
        tensor_check(),
    )

    a_bmm = torch.randn(1, 32, 128, dtype=torch.bfloat16)
    b_bmm = torch.randn(1, 128, 128, dtype=torch.bfloat16)
    fg_bmm = load_op("bmm", "bmm")
    cases["bmm_bf16"] = Case(
        "bmm_bf16",
        lambda: torch.bmm(a_bmm, b_bmm),
        lambda: fg_bmm(a_bmm, b_bmm),
        tensor_check(atol=0.125, rtol=0.02),
    )

    a_mm = torch.randn(1, 2560, dtype=torch.bfloat16)
    b_mm = torch.randn(2560, 4096, dtype=torch.bfloat16)
    fg_mm = load_op("mm", "mm")
    cases["mm_m1_2560x4096_bf16"] = Case(
        "mm_m1_2560x4096_bf16",
        lambda: torch.mm(a_mm, b_mm),
        lambda: fg_mm(a_mm, b_mm),
        tensor_check(atol=1.0, rtol=0.02),
    )

    bias_addmm = torch.randn(4096, dtype=torch.bfloat16)
    fg_addmm = load_op("addmm", "addmm")
    cases["addmm_m1_2560x4096_bf16"] = Case(
        "addmm_m1_2560x4096_bf16",
        lambda: torch.addmm(bias_addmm, a_mm, b_mm),
        lambda: fg_addmm(bias_addmm, a_mm, b_mm),
        tensor_check(atol=1.0, rtol=0.02),
    )

    sort_input = torch.randn(16, 1024, dtype=torch.float32)
    fg_sort = load_op("sort", "sort")
    cases["sort_16x1024_f32"] = Case(
        "sort_16x1024_f32",
        lambda: torch.sort(sort_input, dim=-1),
        lambda: fg_sort(sort_input, dim=-1),
        tuple_check(),
    )

    isin_elements = torch.randint(0, 65536, (131072,), dtype=torch.int32)
    isin_test = torch.randint(0, 65536, (512,), dtype=torch.int32)
    fg_isin = load_op("isin", "isin")
    cases["isin_131k_by_512_i32"] = Case(
        "isin_131k_by_512_i32",
        lambda: torch.isin(isin_elements, isin_test),
        lambda: fg_isin(isin_elements, isin_test),
        tensor_check(),
    )

    scatter_input = torch.randn(128, 1024, dtype=torch.float32)
    scatter_index = torch.randint(0, 1024, (128, 256), dtype=torch.int64)
    scatter_src = torch.randn(128, 256, dtype=torch.float32)
    fg_scatter = load_op("scatter", "scatter")
    cases["scatter_f32"] = Case(
        "scatter_f32",
        lambda: torch.scatter(scatter_input, 1, scatter_index, scatter_src),
        lambda: fg_scatter(scatter_input, 1, scatter_index, scatter_src),
        tensor_check(),
    )
    return cases


def time_call(function: Callable[[], object], target_seconds: float, batches: int):
    for _ in range(3):
        function()
    begin = time.perf_counter_ns()
    function()
    one_ns = max(1, time.perf_counter_ns() - begin)
    iterations = max(1, min(10000, int(target_seconds * 1.0e9 / one_ns)))
    samples = []
    for _ in range(batches):
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - begin) / (iterations * 1000.0))
    return statistics.median(samples), iterations, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="all", help="comma-separated names")
    parser.add_argument("--target-ms", type=float, default=100.0)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--element-size", type=int, default=131072)
    parser.add_argument("--vocab-size", type=int, default=151936)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cases = make_cases(args.element_size, args.vocab_size)
    selected = list(cases) if args.cases == "all" else args.cases.split(",")
    results = []
    for name in selected:
        case = cases[name]
        try:
            expected = case.aten()
            actual = case.gems()
            case.check(actual, expected)
            aten_us, aten_iters, _ = time_call(
                case.aten, args.target_ms / 1000.0, args.batches
            )
            gems_us, gems_iters, _ = time_call(
                case.gems, args.target_ms / 1000.0, args.batches
            )
            result = {
                "case": name,
                "status": "PASS",
                "aten_us": aten_us,
                "gems_us": gems_us,
                "gems_over_aten": gems_us / aten_us,
                "aten_iterations": aten_iters,
                "gems_iterations": gems_iters,
            }
        except Exception as error:  # keep auditing the remaining PR operators
            result = {
                "case": name,
                "status": "ERROR",
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
