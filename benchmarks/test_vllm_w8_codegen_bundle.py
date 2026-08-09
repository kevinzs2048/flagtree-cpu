#!/usr/bin/env python3
"""Validate W8 bundle deployment gates and KleidiAI fallback routing."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = Path(
    os.getenv("VLLM_FL_CHECKOUT", "/home/cix/vllm-plugin-FL-int8")
).resolve()
TRITON_PYTHON = Path(
    os.getenv(
        "TRITON_CPU_PYTHON", ROOT / "ports/triton-cpu-3.7.2/python"
    )
)
VENV_SITE = Path(
    os.getenv(
        "TRITON_CPU_VENV_SITE",
        "/home/cix/venv-fep-e2e/lib/python3.11/site-packages",
    )
)


def load_codegen_module():
    name = "vllm_fl.ops.cpu_int8_triton_codegen"
    source = ROOT / "integrations/vllm/cpu_int8_triton_codegen.py"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        default=(
            ROOT
            / "artifacts/vllm-triton-backend/"
            "libtriton_kai_w8_decode_backend.so"
        ),
    )
    args = parser.parse_args()

    sys.path[:0] = [str(PLUGIN), str(TRITON_PYTHON), str(VENV_SITE)]
    os.environ["FL_CPU_INT8_TRITON_BUNDLE"] = str(args.bundle.resolve())
    os.environ["FL_CPU_INT8_TRITON_LIBRARY"] = str(args.library.resolve())
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    os.environ.setdefault("TRITON_CPU_BACKEND", "1")

    import torch
    from vllm_fl.ops import cpu_int8_kai as kai

    module = load_codegen_module()

    assert not module._m1_codegen_allowed(1024, 1024)
    assert not module._m1_codegen_allowed(2048, 1024)
    production_shapes = (
        (4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072),
        (4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144),
        (6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728),
    )
    assert all(module._m1_codegen_allowed(n, k) for n, k in production_shapes)

    original_runtime_target = module._runtime_target
    module._runtime_target = lambda: (
        "invalid-os",
        "aarch64",
        {"bf16", "dotprod", "i8mm"},
        0,
    )
    try:
        module._load_bundle_manifest()
    except RuntimeError as error:
        assert "target mismatch" in str(error)
    else:
        raise AssertionError("cross-OS bundle was accepted")
    finally:
        module._runtime_target = original_runtime_target

    first_key = next(iter(module._MANIFEST))
    original_digest = module._MANIFEST[first_key]["matrix_sha256"]
    module._MANIFEST[first_key]["matrix_sha256"] = "0" * 64
    try:
        module._verify_shape(*first_key)
    except RuntimeError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("object digest mismatch was accepted")
    finally:
        module._MANIFEST[first_key]["matrix_sha256"] = original_digest

    for n, k in sorted(module._MANIFEST):
        assert module._verify_shape(n, k)
        assert module._create_handle(n, k)

    torch.set_num_threads(1)
    module._ROUTE_THREADS = 1
    for n, k in ((8, 40), (8, 1024)):
        x = torch.randn((1, k), dtype=torch.bfloat16)
        packed = kai._quantize_pack(
            torch.randn((n, k), dtype=torch.bfloat16)
        )
        assert module._create_handle(n, k) is None
        actual = module._make_cpu_linear(packed, n, k, False)(
            x, None, None
        )
        expected = kai.linear_w8a8(x, packed, n, k)
        assert torch.equal(actual, expected)

    n, k = 1024, 1024
    packed = kai._quantize_pack(  # pylint: disable=protected-access
        torch.randn((n, k), dtype=torch.bfloat16)
    )
    handle = module._create_handle(n, k)
    assert handle is not None
    x = torch.randn((1, k), dtype=torch.bfloat16)
    before = module.stats()
    actual = module._make_cpu_linear(packed, n, k, True)(x, None, None)
    expected = kai.linear_w8a8(x, packed, n, k)
    assert torch.equal(actual, expected)
    after = module.stats()
    assert after["m1_codegen_calls"] == before["m1_codegen_calls"]
    assert after["kai_fallback_calls"] == before["kai_fallback_calls"] + 1

    for m in (4, 8, 12, 16):
        x = torch.randn((m, k), dtype=torch.bfloat16)
        before = module.stats()
        actual = module._make_cpu_linear(packed, n, k, True)(x, None, None)
        expected = kai.linear_w8a8(x, packed, n, k)
        assert torch.equal(actual, expected)
        after = module.stats()
        if m == 4:
            assert after["m4_codegen_calls"] == before["m4_codegen_calls"]
            assert after["kai_fallback_calls"] == before["kai_fallback_calls"] + 1
        else:
            assert after[f"m{m}_codegen_calls"] == before[f"m{m}_codegen_calls"] + 1

    accepted_n, accepted_k = 4096, 1024
    accepted_packed = kai._quantize_pack(  # pylint: disable=protected-access
        torch.randn((accepted_n, accepted_k), dtype=torch.bfloat16)
    )
    assert module._create_handle(accepted_n, accepted_k) is not None
    accepted_x = torch.randn((1, accepted_k), dtype=torch.bfloat16)
    before = module.stats()
    actual = module._make_cpu_linear(
        accepted_packed, accepted_n, accepted_k, True
    )(accepted_x, None, None)
    expected = kai.linear_w8a8(
        accepted_x, accepted_packed, accepted_n, accepted_k
    )
    assert torch.equal(actual, expected)
    assert module.stats()["m1_codegen_calls"] == before["m1_codegen_calls"] + 1

    torch.set_num_threads(2)
    module._ROUTE_THREADS = 2
    before = module.stats()
    x = torch.randn((12, k), dtype=torch.bfloat16)
    actual = module._make_cpu_linear(packed, n, k, True)(x, None, None)
    expected = kai.linear_w8a8(x, packed, n, k)
    assert torch.equal(actual, expected)
    assert module.stats()["m12_codegen_calls"] == before["m12_codegen_calls"]

    print(
        "PASS W8 bundle/router gates "
        f"shapes={len(module._MANIFEST)} "
        f"verified={len(module._VERIFIED_SHAPES)} "
        f"handles={len(module._HANDLES)} "
        "target_negative=true digest_negative=true "
        "unsupported_codegen_uses_kleidiai=true "
        "small_m1_uses_kleidiai=true accepted_m1_uses_codegen=true "
        "m4_uses_kleidiai=true "
        "m8_m12_m16_use_codegen=true "
        "high_thread_uses_kleidiai=true"
    )


if __name__ == "__main__":
    main()
