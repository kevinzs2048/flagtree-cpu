#!/usr/bin/env python3
"""Focused vLLM loader-slot integration test for the FlagGems Q4 router."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
sys.path[:0] = [
    str(ROOT / "third_party/FlagGems/src"),
    str(TRITON_PYTHON),
]
if source := os.getenv("VLLM_SOURCE_ROOT"):
    sys.path.insert(2, source)
os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# This test exercises the loader slot directly; unrelated installed vLLM
# plugins would make the result depend on the host environment.
os.environ.setdefault("VLLM_PLUGINS", "__q4_router_test_no_plugins__")

import torch  # noqa: E402
import triton  # noqa: E402


def require_expected_triton() -> None:
    expected = (TRITON_PYTHON / "triton").resolve()
    actual = Path(triton.__file__).resolve()
    if expected not in actual.parents:
        raise RuntimeError(
            "wrong Triton source loaded: "
            f"expected under {expected}, got {actual}. "
            "Put TRITON_CPU_PYTHON at the front of PYTHONPATH before "
            "starting Python."
        )


require_expected_triton()

from flag_gems.runtime.backend._arm.q4 import (  # noqa: E402
    enable_vllm_q4_codegen,
    quantize_q4_0,
    stats,
)
from vllm.model_executor.layers import utils as layer_utils  # noqa: E402


class BareLinear(torch.nn.Module):
    def __init__(self, n: int, k: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn((n, k), dtype=torch.bfloat16), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.randn((n,), dtype=torch.bfloat16), requires_grad=False
        )
        self.prefix = "q4_router_test"


def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    qweight, weight_scale = quantize_q4_0(weight)
    m, k = x.shape
    n = weight.shape[0]
    groups = k // 32
    values = x.float().reshape(m, groups, 32)
    activation_scale = values.abs().amax(dim=-1) / 127.0
    inverse = torch.where(
        activation_scale != 0,
        1.0 / activation_scale,
        torch.zeros_like(activation_scale),
    )
    qactivation = (
        (values * inverse[..., None]).clamp(-127, 127).round().to(torch.int8)
    )
    result = torch.zeros((m, n), dtype=torch.float32)
    for group in range(groups):
        begin = group * 32
        dot = (
            qactivation[:, group].to(torch.int32)
            @ qweight[:, begin : begin + 32].to(torch.int32).T
        ).float()
        result += (
            dot
            * activation_scale[:, group, None].to(torch.float16).float()
            * weight_scale[:, group].float()[None, :]
        )
    return result.to(torch.bfloat16)


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(811)
    enable_vllm_q4_codegen(verbose=False)
    layer = BareLinear(64, 128)
    weight = layer.weight.detach().clone()
    bias = layer.bias.detach().clone()
    layer_utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)
    assert callable(layer.cpu_linear)
    assert layer.weight.numel() == 0
    for m in (1, 3, 4, 12, 16, 20, 31, 32, 47):
        x = torch.randn((m, 128), dtype=torch.bfloat16)
        output = layer.cpu_linear(x, layer.weight, bias)
        assert output.shape == (m, 64)
        assert output.dtype == torch.bfloat16
        expected = reference(x, weight) + bias
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    result = stats()
    assert result["prepared_linears"] == 1
    assert result["decode_codegen_calls"] == 2
    assert result["codegen_prefill_calls"] == 7
    compile_x = torch.randn((1, 128), dtype=torch.bfloat16)
    compile_ref = layer.cpu_linear(compile_x, layer.weight, bias)
    compiled_slot = torch.compile(
        lambda value: layer.cpu_linear(value, layer.weight, bias),
        backend="eager",
        fullgraph=True,
    )
    assert torch.equal(compiled_slot(compile_x), compile_ref)
    print(
        "PASS vLLM FlagGems Q4 loader/router",
        result,
        {"eager_direct": True, "compile_fullgraph": True},
    )


if __name__ == "__main__":
    main()
