"""Correctness checks for register-bounded Arm CPU Triton schedules."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_WORKTREE_FLAGGEMS = (
    Path(__file__).resolve().parents[1] / "third_party" / "FlagGems" / "src"
)
sys.path.insert(0, str(_WORKTREE_FLAGGEMS))

import torch


sys.meta_path = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ != "_flag_gems_editable"
]


def arm_op(module: str, name: str):
    full_name = f"flag_gems.runtime.backend._arm.ops.{module}"
    return getattr(importlib.import_module(full_name), name)


def test_rolled_elementwise_bf16_tails():
    for size in (1, 3, 63, 64, 65, 4097, 131073):
        torch.manual_seed(size)
        x = torch.randn(size, dtype=torch.bfloat16)
        y = torch.randn(size, dtype=torch.bfloat16)
        condition = torch.rand(size) > 0.5

        torch.testing.assert_close(arm_op("lt", "lt")(x, y), torch.lt(x, y))
        torch.testing.assert_close(
            arm_op("where", "where_self")(condition, x, y),
            torch.where(condition, x, y),
        )
        torch.testing.assert_close(
            arm_op("where", "where_scalar_self")(condition, 0.25, y),
            torch.where(condition, 0.25, y),
        )
        torch.testing.assert_close(
            arm_op("masked_fill", "masked_fill")(x, condition, -3.0),
            torch.masked_fill(x, condition, -3.0),
        )
        torch.testing.assert_close(
            arm_op("sub", "sub")(x, y, alpha=0.75),
            torch.sub(x, y, alpha=0.75),
        )
        torch.testing.assert_close(
            arm_op("div", "true_divide")(x, 1.7),
            torch.true_divide(x, 1.7),
        )
        torch.testing.assert_close(
            arm_op("pow", "pow_tensor_scalar")(x, 2.0),
            torch.pow(x, 2.0),
        )


def test_rolled_reduction_tails_and_nan():
    for size in (1, 15, 16, 17, 151937):
        torch.manual_seed(size + 4000)
        x = torch.randn(size, dtype=torch.bfloat16)
        truth = torch.rand(size) > 0.2

        torch.testing.assert_close(arm_op("min", "min")(x), torch.min(x))
        torch.testing.assert_close(arm_op("all", "all")(truth), torch.all(truth))
        torch.testing.assert_close(arm_op("any", "any")(truth), torch.any(truth))

        x[size // 2] = float("nan")
        assert torch.isnan(arm_op("min", "min")(x))


def test_empty_boolean_reductions():
    empty = torch.empty(0, dtype=torch.bool)
    assert bool(arm_op("all", "all")(empty))
    assert not bool(arm_op("any", "any")(empty))


def test_ordinary_swiglu_bf16_semantics():
    module = importlib.import_module(
        "flag_gems.runtime.backend._arm.ops.silu_and_mul"
    )
    for size in (15, 16, 17, 6913):
        torch.manual_seed(size + 7000)
        gate = torch.randn(size, dtype=torch.bfloat16)
        up = torch.randn(size, dtype=torch.bfloat16)
        out = torch.empty_like(gate)
        module._swiglu_ordinary_kernel[(1,)](
            gate,
            up,
            out,
            size,
            BLOCK_SIZE=module._SWIGLU_TILE,
            num_warps=1,
            num_stages=1,
        )
        torch.testing.assert_close(out, torch.nn.functional.silu(gate) * up)


def test_conservative_registration_policy():
    arm_backend = importlib.import_module("flag_gems.runtime.backend._arm")
    arm_ops = importlib.import_module("flag_gems.runtime.backend._arm.ops")
    large_only = {
        "all",
        "any",
        "masked_fill",
        "pow_tensor_scalar",
        "sub",
        "true_divide",
        "where_self_out",
    }
    assert large_only <= set(arm_backend.CUSTOMIZED_UNUSED_OPS)
    globally_unsafe = {
        "_int_mm",
        "argmax",
        "mm",
        "quantized_linear_dynamic",
        "silu_and_mul",
    }
    assert globally_unsafe.isdisjoint(arm_ops._ARM_DEFAULT_OVERRIDES)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    test_rolled_elementwise_bf16_tails()
    test_rolled_reduction_tails_and_nan()
    test_empty_boolean_reductions()
    test_ordinary_swiglu_bf16_semantics()
    test_conservative_registration_policy()
    print("PASS FlagGems Arm rolled-op correctness")


if __name__ == "__main__":
    main()
