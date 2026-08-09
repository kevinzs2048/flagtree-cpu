#!/usr/bin/env python3
"""Run direct-call Arm native parity gates for ordinary Triton kernels.

The gate deliberately keeps compilation, Python dispatch, packing, and model
integration out of matrix-only timing.  Correctness and timing are performed
by the existing C++ harnesses on identical addresses and packed blobs.  The
runner only compiles the requested Triton specialization, invokes the native
harness, parses its machine-readable key/value output, and writes one JSON and
Markdown report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "operator-native-gate"
PORT_PYTHON = ROOT / "ports" / "triton-cpu-3.7.2" / "python"
DEFAULT_PYTHON = Path("/home/cix/venv-fep-e2e/bin/python")
TORCH_SITE = Path("/home/cix/venv-fep-e2e/lib/python3.11/site-packages")

W8_SHAPES = (
    (1024, 1024),
    (1024, 2048),
    (1024, 3072),
    (1024, 4096),
    (2048, 1024),
    (1024, 6144),
    (3072, 1024),
    (2048, 4096),
    (2048, 2048),
    (2048, 12288),
    (6144, 2048),
    (2560, 1024),
    (2560, 4096),
    (2560, 9728),
    (4096, 2560),
    (9728, 2560),
    (2560, 6144),
    (2560, 19456),
)

W4_SHAPES = (
    (1024, 1024),
    (1024, 2048),
    (1024, 3072),
    (1024, 4096),
    # Qwen3-1.7B TP1 projections: fused QKV, O, fused gate/up, and down.
    (2048, 4096),
    (2048, 2048),
    (2048, 12288),
    (6144, 2048),
    (2560, 1024),
    (2560, 4096),
    (2560, 9728),
    (4096, 2560),
    (9728, 2560),
)

QK_RMS_SHAPES = (
    (8, 8, 64),
    (16, 8, 64),
    (16, 8, 128),
    (32, 8, 128),
    (16, 8, 256),
    (32, 8, 256),
)

SWIGLU_QUANT_SHAPES = (1024, 2048, 3072, 4096, 6144, 8192)

QKV_PIPELINE_SHAPES = (
    (1024, 4096, "qwen-0.6b"),
    (2048, 4096, "qwen-1.7b"),
    (2560, 6144, "qwen-4b"),
)

KEY_VALUE = re.compile(r"^([A-Za-z0-9_]+)=([^\s]+)$")


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> str:
    print(f"+ {command_text(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{command_text(command)}"
        )
    return completed.stdout


def parse_metrics(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in output.splitlines():
        match = KEY_VALUE.match(line.strip())
        if not match:
            continue
        key, raw = match.groups()
        value = raw[:-1] if raw.endswith("x") else raw
        if value.lower() in ("true", "false"):
            metrics[key] = value.lower() == "true"
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = raw
    return metrics


def classify(ratio: float) -> tuple[str, bool]:
    if ratio <= 0.98:
        return "sweet", True
    if ratio <= 1.03:
        return "parity", True
    if ratio <= 1.05:
        return "borderline", False
    return "native_wins", False


def zero_mismatches(value: Any) -> bool:
    text = str(value)
    numerator = text.split("/", 1)[0]
    try:
        return float(numerator) == 0.0
    except ValueError:
        return False


def iterations_for(k: int, n: int) -> int:
    # Roughly 3e8 KxN scalar products per timed batch.  The C++ harness then
    # takes the median of alternating-order batches.
    return max(12, min(200, round(300_000_000 / (k * n))))


def triton_environment(cache: Path) -> dict[str, str]:
    env = os.environ.copy()
    inherited = env.get("PYTHONPATH")
    python_path = [str(PORT_PYTHON), str(TORCH_SITE)]
    if inherited:
        python_path.append(inherited)
    env.update(
        {
            "PYTHONPATH": ":".join(python_path),
            "TRITON_BACKENDS_IN_TREE": "1",
            "TRITON_CPU_BACKEND": "1",
            "TRITON_DISABLE_LINE_INFO": "1",
            "TRITON_CACHE_DIR": str(cache),
            "OMP_NUM_THREADS": "1",
        }
    )
    return env


def compiled_object(cache: Path, symbol: str) -> Path:
    objects = sorted(cache.rglob(f"{symbol}.so"))
    if len(objects) != 1:
        raise RuntimeError(
            f"expected one {symbol}.so below {cache}, found {len(objects)}"
        )
    return objects[0]


def same_backend_commands(cpu: int, rebuild: bool) -> list[list[str]]:
    output = ROOT / "artifacts" / "same-backend-frontend-ab"
    if rebuild or not (output / "bench").exists():
        return [["bash", "benchmarks/same_backend_frontend_ab/run.sh"]]
    common = [
        str(output / "c_frontend.so"),
        str(output / "c_acle.so"),
        str(output / "triton_rms.so"),
        str(output / "triton_rope.so"),
        str(output / "triton_w8.so"),
    ]
    return [
        ["taskset", "-c", str(cpu), str(output / "bench"), *common],
        ["taskset", "-c", str(cpu), str(output / "correctness"), *common],
    ]


def run_same_backend(cpu: int, rebuild: bool) -> list[dict[str, Any]]:
    combined = ""
    env = os.environ.copy()
    env["SAME_BACKEND_CPU"] = str(cpu)
    for command in same_backend_commands(cpu, rebuild):
        combined += run(command, env=env)
    metrics = parse_metrics(combined)
    rows = []
    definitions = (
        ("rmsnorm_bf16_n1024", "rms_triton_us", "rms_acle_c_us",
         "rms_triton_over_acle", "rms_acle_mismatch"),
        ("rope_bf16_q16_k8_d128", "rope_triton_us", "rope_acle_c_us",
         "rope_triton_over_acle", "rope_acle_mismatch"),
        ("w8_kai_k1024_n3072", "w8_triton_us", "w8_acle_c_us",
         "w8_triton_over_acle", None),
    )
    for name, triton_key, native_key, ratio_key, mismatch_key in definitions:
        ratio = float(metrics[ratio_key])
        status, eligible = classify(ratio)
        accuracy_ok = (
            zero_mismatches(metrics.get(mismatch_key, 0.0))
            if mismatch_key
            else float(metrics.get("w8_acle_max_abs", 1.0)) == 0.0
        )
        rows.append(
            {
                "operator": name,
                "reference": "ACLE C",
                "triton_us": metrics[triton_key],
                "native_us": metrics[native_key],
                "ratio": ratio,
                "accuracy_ok": accuracy_ok,
                "status": status,
                "eligible": eligible and accuracy_ok,
            }
        )
    return rows


def run_quantized_shape(
    kind: str,
    k: int,
    n: int,
    cpu: int,
    batches: int,
    run_dir: Path,
    python: Path,
) -> dict[str, Any]:
    cache = run_dir / "cache" / f"{kind}-k{k}-n{n}"
    cache.mkdir(parents=True, exist_ok=True)
    if kind == "w8":
        generator = ROOT / "benchmarks" / "bench_w8_kleidiai_layout_codegen.py"
        symbol = "_kai_w8_layout_pointer_kernel"
        compile_args = [
            str(python), str(generator), "--compile-only", "--mode", "pointer",
            "--block-n", "4", "--unroll", "2", "--k", str(k), "--n", str(n),
        ]
        bench = ROOT / "artifacts" / "bench_w8_kleidiai_layout"
    else:
        generator = ROOT / "benchmarks" / "bench_w4_kleidiai_layout_codegen.py"
        symbol = "_kai_w4_layout_split_kernel"
        unroll = 4 if k >= 4096 else 1
        compile_args = [
            str(python), str(generator), "--compile-only", "--unroll", str(unroll),
            "--k", str(k), "--n", str(n),
        ]
        bench = ROOT / "artifacts" / "bench_w4_kleidiai_layout"
    if not bench.exists():
        raise RuntimeError(
            f"missing {bench}; run benchmarks/cpp_wrapper/build.sh first"
        )
    compile_output = run(compile_args, env=triton_environment(cache))
    shared_object = compiled_object(cache, symbol)
    iterations = iterations_for(k, n)
    command = [
        "taskset", "-c", str(cpu), str(bench), str(shared_object), str(k),
        str(n), str(iterations), str(batches),
    ]
    if kind == "w8":
        command.extend(("4", "pointer"))
    bench_output = run(command)
    metrics = parse_metrics(bench_output)
    ratio = float(metrics["triton_over_kleidiai"])
    status, eligible = classify(ratio)
    max_abs = float(metrics["max_abs_error"])
    accuracy_ok = max_abs <= (2.0e-6 if kind == "w8" else 3.0e-5)
    audit = parse_metrics(compile_output)
    return {
        "operator": f"{kind}_decode_k{k}_n{n}",
        "reference": "KleidiAI v1.24",
        "k": k,
        "n": n,
        "iterations": iterations,
        "triton_us": metrics["triton_kernel_us"],
        "native_us": metrics["kleidiai_kernel_us"],
        "ratio": ratio,
        "max_abs_error": max_abs,
        "accuracy_ok": accuracy_ok,
        "status": status,
        "eligible": eligible and accuracy_ok,
        "asm_sdot": audit.get("asm_sdot"),
        "asm_addp": audit.get("asm_addp"),
        "stack_load_store": audit.get("stack_load_store"),
        "external_calls": audit.get("external_calls"),
        "shared_object": str(shared_object),
    }


def run_qk_rmsnorm_shapes(
    cpu: int,
    batches: int,
    run_dir: Path,
    python: Path,
    quick: bool,
) -> list[dict[str, Any]]:
    cache = run_dir / "cache" / "qk-rmsnorm"
    cache.mkdir(parents=True, exist_ok=True)
    generator = ROOT / "benchmarks" / "bench_qk_rmsnorm_codegen.py"
    compile_output = run(
        [
            str(python), "-S", str(generator), "--compile-only", "--threads",
            "1", "--q-heads", "16", "--kv-heads", "8", "--head-dim", "128",
        ],
        env=triton_environment(cache),
    )
    shared_object = compiled_object(
        cache, "_qk_rms_norm_contiguous_kernel"
    )
    bench = ROOT / "artifacts" / "bench_qk_rmsnorm_native"
    if not bench.exists():
        raise RuntimeError(
            f"missing {bench}; run benchmarks/cpp_wrapper/build.sh first"
        )
    audit = parse_metrics(compile_output)
    shapes = QK_RMS_SHAPES[:3] if quick else QK_RMS_SHAPES
    rows: list[dict[str, Any]] = []
    for q_heads, kv_heads, head_dim in shapes:
        output = run(
            [
                "taskset", "-c", str(cpu), str(bench), str(shared_object),
                str(q_heads), str(kv_heads), str(head_dim), "10000",
                str(batches),
            ]
        )
        metrics = parse_metrics(output)
        ratio = float(metrics["triton_over_acle"])
        status, eligible = classify(ratio)
        accuracy_ok = zero_mismatches(metrics["mismatches"])
        rows.append(
            {
                "operator": (
                    f"qk_rmsnorm_hq{q_heads}_hkv{kv_heads}_d{head_dim}"
                ),
                "reference": "fused ACLE C",
                "triton_us": metrics["triton_direct_us"],
                "native_us": metrics["acle_fused_us"],
                "ratio": ratio,
                "max_bf16_ulp": metrics["max_bf16_ulp"],
                "accuracy_cases": metrics.get("accuracy_cases", 1),
                "accuracy_ok": accuracy_ok,
                "status": status,
                "eligible": eligible and accuracy_ok,
                "asm_lines": audit.get("asm_lines"),
                "stack_load_store": audit.get("stack_load_store"),
                "external_calls": audit.get("external_calls"),
                "shared_object": str(shared_object),
            }
        )
    return rows


def run_swiglu_quant_shapes(
    cpu: int,
    batches: int,
    run_dir: Path,
    python: Path,
    quick: bool,
) -> list[dict[str, Any]]:
    bench = ROOT / "artifacts" / "bench_swiglu_quant_native"
    if not bench.exists():
        raise RuntimeError(
            f"missing {bench}; run benchmarks/cpp_wrapper/build.sh first"
        )
    shapes = SWIGLU_QUANT_SHAPES[:3] if quick else SWIGLU_QUANT_SHAPES
    rows: list[dict[str, Any]] = []
    for elements in shapes:
        cache = run_dir / "cache" / f"swiglu-quant-n{elements}"
        cache.mkdir(parents=True, exist_ok=True)
        compile_output = run(
            [
                str(python), "-S",
                str(ROOT / "benchmarks" / "bench_swiglu_quant_codegen.py"),
                "--compile-only", "--threads", "1", "--elements",
                str(elements),
            ],
            env=triton_environment(cache),
        )
        shared_object = compiled_object(
            cache, "_swiglu_quantize_w8_rne_kernel"
        )
        iterations = max(1000, min(8000, round(9_000_000 / elements)))
        output = run(
            [
                "taskset", "-c", str(cpu), str(bench),
                str(shared_object), str(elements), str(iterations),
                str(batches),
            ]
        )
        metrics = parse_metrics(output)
        audit = parse_metrics(compile_output)
        ratio = float(metrics["triton_over_acle"])
        status, eligible = classify(ratio)
        accuracy_ok = (
            zero_mismatches(metrics["bf16_mismatch"])
            and zero_mismatches(metrics["q_mismatch"])
            and bool(metrics["scale_bit_exact"])
        )
        rows.append(
            {
                "operator": f"swiglu_w8_quant_n{elements}",
                "reference": "fused ACLE C",
                "n": elements,
                "iterations": iterations,
                "triton_us": metrics["triton_direct_us"],
                "native_us": metrics["acle_fused_us"],
                "ratio": ratio,
                "accuracy_ok": accuracy_ok,
                "bf16_exact": zero_mismatches(metrics["bf16_mismatch"]),
                "q_exact": zero_mismatches(metrics["q_mismatch"]),
                "scale_bit_exact": metrics["scale_bit_exact"],
                "accuracy_cases": metrics.get("accuracy_cases", 1),
                "status": status,
                "eligible": eligible and accuracy_ok,
                "asm_lines": audit.get("asm_lines"),
                "stack_load_store": audit.get("stack_load_store"),
                "external_calls": audit.get("external_calls"),
                "shared_object": str(shared_object),
            }
        )
    return rows


def run_qkv_pipeline_shapes(
    cpu: int,
    batches: int,
    run_dir: Path,
    python: Path,
    quick: bool,
) -> list[dict[str, Any]]:
    bench = ROOT / "artifacts" / "bench_w8_kai_bf16_decode_pipeline"
    if not bench.exists():
        raise RuntimeError(
            f"missing {bench}; run benchmarks/cpp_wrapper/build.sh first"
        )
    shapes = QKV_PIPELINE_SHAPES[:2] if quick else QKV_PIPELINE_SHAPES
    rows: list[dict[str, Any]] = []
    for k, n, model in shapes:
        shape_dir = run_dir / "cache" / f"qkv-pipeline-k{k}-n{n}"
        pack_cache = shape_dir / "pack"
        matrix_cache = shape_dir / "matrix"
        pack_cache.mkdir(parents=True, exist_ok=True)
        matrix_cache.mkdir(parents=True, exist_ok=True)
        pack_output = run(
            [
                str(python),
                str(ROOT / "benchmarks" /
                    "bench_w8_kai_bf16_lhs_pack_codegen.py"),
                "--compile-only", "--m", "1", "--mr", "1", "--k",
                str(k),
            ],
            env=triton_environment(pack_cache),
        )
        matrix_output = run(
            [
                str(python),
                str(ROOT / "benchmarks" /
                    "bench_w8_kleidiai_layout_codegen.py"),
                "--compile-only", "--mode", "pointer", "--block-n", "4",
                "--unroll", "2", "--output-bf16", "--k", str(k), "--n",
                str(n),
            ],
            env=triton_environment(matrix_cache),
        )
        pack_object = compiled_object(
            pack_cache, "_pack_lhs_qai8dxp_bf16_kernel"
        )
        matrix_object = compiled_object(
            matrix_cache, "_kai_w8_layout_pointer_kernel"
        )
        iterations = iterations_for(k, n)
        output = run(
            [
                "taskset", "-c", str(cpu), str(bench), str(pack_object),
                str(matrix_object), str(k), str(n), str(iterations),
                str(batches),
            ]
        )
        metrics = parse_metrics(output)
        matrix_audit = parse_metrics(matrix_output)
        ratio = float(metrics["triton_over_kleidiai"])
        status, eligible = classify(ratio)
        accuracy_ok = bool(metrics["lhs_pack_bit_exact"]) and bool(
            metrics["bf16_output_bit_exact"]
        )
        # The pack generator emits a JSON audit.  Keep the raw values in the
        # report without making report generation depend on warning-free
        # stdout from PyTorch/FlagGems imports.
        pack_audit: dict[str, Any] = {}
        json_begin = pack_output.find("{")
        if json_begin >= 0:
            try:
                pack_audit = json.loads(pack_output[json_begin:])
            except json.JSONDecodeError:
                pack_audit = {}
        rows.append(
            {
                "operator": f"fused_qkv_w8_{model}_k{k}_n{n}",
                "reference": "KleidiAI BF16 decode pipeline",
                "k": k,
                "n": n,
                "iterations": iterations,
                "triton_us": metrics["triton_pipeline_us"],
                "native_us": metrics["kleidiai_pipeline_us"],
                "ratio": ratio,
                "accuracy_ok": accuracy_ok,
                "lhs_pack_bit_exact": metrics["lhs_pack_bit_exact"],
                "bf16_output_bit_exact": metrics["bf16_output_bit_exact"],
                "status": status,
                "eligible": eligible and accuracy_ok,
                "pack_asm_lines": pack_audit.get("asm_lines"),
                "pack_fcvtns": pack_audit.get("fcvtns"),
                "pack_stack_load_store": pack_audit.get(
                    "stack_load_store"
                ),
                "matrix_sdot": matrix_audit.get("asm_sdot"),
                "matrix_addp": matrix_audit.get("asm_addp"),
                "matrix_stack_load_store": matrix_audit.get(
                    "stack_load_store"
                ),
                "matrix_external_calls": matrix_audit.get("external_calls"),
                "pack_shared_object": str(pack_object),
                "matrix_shared_object": str(matrix_object),
            }
        )
    return rows


def write_report(report: dict[str, Any], output_json: Path) -> Path:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    output_md = output_json.with_suffix(".md")
    lines = [
        "# Ordinary-Triton native microbenchmark gate",
        "",
        f"- Date: {report['date']}",
        f"- CPU: {report['cpu']}",
        "- Timing: direct shared-object call, packing and Python excluded",
        "- Eligibility: accuracy passes and Triton/native <= 1.03",
        "",
        "| Operator | Native reference | Triton us | Native us | Ratio | Accuracy | Class | Eligible |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['operator']} | {row['reference']} | "
            f"{float(row['triton_us']):.4f} | {float(row['native_us']):.4f} | "
            f"{float(row['ratio']):.4f}x | "
            f"{'PASS' if row['accuracy_ok'] else 'FAIL'} | {row['status']} | "
            f"{'yes' if row['eligible'] else 'no'} |"
        )
    output_md.write_text("\n".join(lines) + "\n")
    return output_md


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suites", default="same,w8,w4,qk-rms,swiglu-quant,qkv-pipeline",
        help=("comma-separated subset of same,w8,w4,qk-rms,swiglu-quant,"
              "qkv-pipeline"),
    )
    parser.add_argument("--cpu", type=int, default=11)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rebuild-same-backend", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suites = {item.strip() for item in args.suites.split(",") if item.strip()}
    unknown = suites - {
        "same", "w8", "w4", "qk-rms", "swiglu-quant", "qkv-pipeline"
    }
    if unknown:
        parser.error(f"unknown suites: {sorted(unknown)}")
    if not args.python.exists():
        parser.error(f"Python executable does not exist: {args.python}")

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = ARTIFACT_ROOT / stamp
    results: list[dict[str, Any]] = []
    if "same" in suites:
        results.extend(run_same_backend(args.cpu, args.rebuild_same_backend))
    if "w8" in suites:
        shapes = W8_SHAPES[:4] if args.quick else W8_SHAPES
        for k, n in shapes:
            results.append(
                run_quantized_shape(
                    "w8", k, n, args.cpu, args.batches, run_dir, args.python
                )
            )
    if "w4" in suites:
        shapes = W4_SHAPES[:4] if args.quick else W4_SHAPES
        for k, n in shapes:
            results.append(
                run_quantized_shape(
                    "w4", k, n, args.cpu, args.batches, run_dir, args.python
                )
            )
    if "qk-rms" in suites:
        results.extend(
            run_qk_rmsnorm_shapes(
                args.cpu, args.batches, run_dir, args.python, args.quick
            )
        )
    if "swiglu-quant" in suites:
        results.extend(
            run_swiglu_quant_shapes(
                args.cpu, args.batches, run_dir, args.python, args.quick
            )
        )
    if "qkv-pipeline" in suites:
        results.extend(
            run_qkv_pipeline_shapes(
                args.cpu, args.batches, run_dir, args.python, args.quick
            )
        )

    report = {
        "date": dt.datetime.now().astimezone().isoformat(),
        "cpu": args.cpu,
        "suites": sorted(suites),
        "results": results,
    }
    output = args.output or run_dir / "results.json"
    report_md = write_report(report, output)
    print(f"JSON report: {output}")
    print(f"Markdown report: {report_md}")
    failures = [row for row in results if not row["accuracy_ok"]]
    if failures:
        print(f"ERROR: {len(failures)} accuracy failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
