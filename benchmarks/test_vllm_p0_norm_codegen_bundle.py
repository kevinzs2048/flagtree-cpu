#!/usr/bin/env python3
"""Deployment and exactness gates for the generated vLLM P0 Norm bundle."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        default=(
            ROOT
            / "artifacts/vllm-triton-backend/libtriton_p0_norm_backend.so"
        ),
    )
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    library = args.library.resolve()
    os.environ["FL_CPU_P0_TRITON_OPS"] = "rms,fused_rms,rope,swiglu"

    name = "_test_cpu_p0_triton_codegen"
    source = ROOT / "integrations/vllm/cpu_p0_triton_codegen.py"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    module._validate_bundle(bundle)  # pylint: disable=protected-access
    backend = module._P0Backend(library, bundle)  # pylint: disable=protected-access
    torch.manual_seed(20260804)
    try:
        for rows, cols in ((1, 1024), (16, 128), (8, 128)):
            x = torch.randn(rows, cols, dtype=torch.bfloat16)
            weight = torch.randn(cols, dtype=torch.bfloat16)
            out = torch.empty_like(x)
            backend.rms(x, weight, out)
            xf = x.float()
            rrms = torch.rsqrt(
                (xf * xf).mean(dim=-1, keepdim=True) + 1.0e-6
            )
            expected = ((xf * rrms).to(torch.bfloat16) * weight).to(
                torch.bfloat16
            )
            if not torch.equal(out, expected):
                raise AssertionError(f"RMSNorm mismatch for {(rows, cols)}")

        x = torch.randn(1, 1024, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(1024, dtype=torch.bfloat16)
        x_initial = x.clone()
        residual_initial = residual.clone()
        backend.fused_rms(x, residual, weight)
        updated = x_initial.float() + residual_initial.float()
        expected_residual = updated.to(torch.bfloat16)
        rrms = torch.rsqrt(
            (updated * updated).mean(dim=-1, keepdim=True) + 1.0e-6
        )
        expected_x = ((updated * rrms).to(torch.bfloat16) * weight).to(
            torch.bfloat16
        )
        if not torch.equal(residual, expected_residual):
            raise AssertionError("fused RMSNorm residual mismatch")
        if not torch.equal(x, expected_x):
            raise AssertionError("fused RMSNorm output mismatch")

        query = torch.randn(16, 128, dtype=torch.bfloat16)
        key = torch.randn(8, 128, dtype=torch.bfloat16)
        positions = torch.tensor([17], dtype=torch.int64)
        cos_sin_cache = torch.randn(32, 128, dtype=torch.bfloat16)
        cosine = cos_sin_cache[17, :64]
        sine = cos_sin_cache[17, 64:]
        query_initial = query.clone()
        key_initial = key.clone()
        backend.rope(positions, query, key, cos_sin_cache)

        def rope_reference(value: torch.Tensor) -> torch.Tensor:
            first, second = value[:, :64].float(), value[:, 64:].float()
            first_out = first * cosine.float() - second * sine.float()
            second_out = second * cosine.float() + first * sine.float()
            return torch.cat((first_out, second_out), dim=-1).to(
                torch.bfloat16
            )

        if not torch.equal(query, rope_reference(query_initial)):
            raise AssertionError("RoPE query mismatch")
        if not torch.equal(key, rope_reference(key_initial)):
            raise AssertionError("RoPE key mismatch")

        joined = torch.randn(1, 6144, dtype=torch.bfloat16) * 0.2
        swiglu = torch.empty(1, 3072, dtype=torch.bfloat16)
        backend.swiglu(joined, swiglu)
        expected_swiglu = (
            torch.nn.functional.silu(joined[:, :3072])
            * joined[:, 3072:]
        ).to(torch.bfloat16)
        if not torch.equal(swiglu, expected_swiglu):
            raise AssertionError("SwiGLU mismatch")
    finally:
        backend.close()

    with tempfile.TemporaryDirectory(prefix="p0-norm-bundle-test-") as temp:
        corrupt = Path(temp) / "bundle"
        shutil.copytree(bundle, corrupt)
        object_path = (
            corrupt
            / "rms-m1-n1024-e1e-6/_rms_norm_aot_kernel.so"
        )
        object_path.chmod(0o644)
        with object_path.open("ab") as stream:
            stream.write(b"digest-corruption-test")
        try:
            module._validate_bundle(corrupt)  # pylint: disable=protected-access
        except RuntimeError as exc:
            if "digest mismatch" not in str(exc):
                raise
        else:
            raise AssertionError("corrupt P0 object was accepted")

    print(
        "PASS P0 Norm deployment gates\n"
        "ordinary_triton_objects=6\n"
        "rms_shapes=3\n"
        "fused_rms_shapes=1\n"
        "rope_shapes=1\n"
        "swiglu_shapes=1\n"
        "bit_exact=True\n"
        "digest_rejection=True"
    )


if __name__ == "__main__":
    main()
