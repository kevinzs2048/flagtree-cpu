#!/usr/bin/env python3
"""Paired plugin-layer microbench for generated M1/M8/M12/M16 versus KAI."""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = Path(
    os.getenv("VLLM_FL_CHECKOUT", "/home/cix/vllm-plugin-FL-int8")
).resolve()
os.environ.setdefault(
    "FL_CPU_INT8_TRITON_LIBRARY",
    str(
        ROOT
        / "artifacts/vllm-triton-backend/"
        "libtriton_kai_w8_decode_backend.so"
    ),
)
sys.path.insert(0, str(PLUGIN))

from vllm_fl.ops import cpu_int8_kai as kai  # noqa: E402


def load_codegen_module():
    source = ROOT / "integrations/vllm/cpu_int8_triton_codegen.py"
    module_name = "vllm_fl.ops.cpu_int8_triton_codegen"
    spec = importlib.util.spec_from_file_location(
        module_name, source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def timed_us(function, iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    return (time.perf_counter_ns() - start) / iterations / 1000.0


def paired_medians(first, second, warmup: int, iterations: int, batches: int):
    for _ in range(warmup):
        first()
        second()
    first_samples = []
    second_samples = []
    for batch in range(batches):
        if batch % 2 == 0:
            first_samples.append(timed_us(first, iterations))
            second_samples.append(timed_us(second, iterations))
        else:
            second_samples.append(timed_us(second, iterations))
            first_samples.append(timed_us(first, iterations))
    return statistics.median(first_samples), statistics.median(second_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, choices=(1, 4, 8, 12, 16), default=16)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batches", type=int, default=21)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=(
            Path(os.environ["FL_CPU_INT8_TRITON_BUNDLE"])
            if "FL_CPU_INT8_TRITON_BUNDLE" in os.environ
            else None
        ),
        help="exact target/codegen-key directory printed by the AOT builder",
    )
    parser.add_argument(
        "--compiled-dynamic",
        action="store_true",
        help="reuse one fullgraph dynamic compile across generated/fallback M",
    )
    args = parser.parse_args()
    if args.k % 32 or args.n % 4:
        raise ValueError("this benchmark requires M=1/16, K%32=0, N%4=0")
    if args.bundle is None:
        raise ValueError(
            "--bundle or FL_CPU_INT8_TRITON_BUNDLE is required"
        )
    os.environ["FL_CPU_INT8_TRITON_BUNDLE"] = str(args.bundle.resolve())

    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    codegen = load_codegen_module()
    codegen._ROUTE_THREADS = args.torch_threads  # pylint: disable=protected-access
    torch.manual_seed(11731)
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16)
    weight = torch.randn((args.n, args.k), dtype=torch.bfloat16)
    packed = kai._quantize_pack(weight)  # pylint: disable=protected-access
    handle = codegen._create_handle(args.n, args.k)  # pylint: disable=protected-access
    if handle is None:
        raise RuntimeError("shape is absent from the generated bundle")
    production_linear = codegen._make_cpu_linear(  # pylint: disable=protected-access
        packed, args.n, args.k, True
    )

    if args.compiled_dynamic:
        def operation(values):
            return production_linear(values, None, None)

        compiled = torch.compile(operation, fullgraph=True, dynamic=True)
        for dynamic_m in (1, 4, 8, 12, 16, 3, 1):
            dynamic_x = torch.randn(
                (dynamic_m, args.k), dtype=torch.bfloat16
            )
            generated = compiled(dynamic_x)
            reference = kai.linear_w8a8(
                dynamic_x, packed, args.n, args.k
            )
            if not torch.equal(generated, reference):
                mismatch = torch.nonzero(
                    generated.view(torch.uint16)
                    != reference.view(torch.uint16)
                )
                raise AssertionError(
                    f"compiled dynamic M={dynamic_m} KAI mismatch "
                    f"count={mismatch.shape[0]} first={mismatch[:16].tolist()}"
                )
        print(
            "PASS vLLM W8 compiled dynamic router "
            f"K={args.k} N={args.n} shapes=1,4,8,12,16,3,1",
            flush=True,
        )

    def run_codegen():
        return production_linear(x, None, None)

    def run_kai():
        return kai.linear_w8a8(x, packed, args.n, args.k)

    generated = run_codegen()
    reference = run_kai()
    if not torch.equal(generated, reference):
        mismatch = torch.nonzero(
            generated.view(torch.uint16) != reference.view(torch.uint16)
        ).flatten()
        raise AssertionError(
            f"KAI mismatch count={mismatch.numel()} first={mismatch[:16].tolist()}"
        )

    generated_us, kai_us = paired_medians(
        run_codegen,
        run_kai,
        args.warmup,
        args.iterations,
        args.batches,
    )
    print(
        f"PASS vLLM W8 plugin M={args.m} K={args.k} N={args.n}\n"
        f"generated_us={generated_us:.6f}\n"
        f"kleidiai_us={kai_us:.6f}\n"
        f"generated_over_kleidiai={generated_us / kai_us:.8f}x\n"
        "bf16_output_bit_exact=true\n"
        "weight_pack_excluded=true"
    )


if __name__ == "__main__":
    main()
