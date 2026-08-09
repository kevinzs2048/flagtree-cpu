#!/usr/bin/env python3
"""Real vLLM Qwen3 W8 decode A/B: generated backend versus KleidiAI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = Path(
    os.getenv("VLLM_FL_CHECKOUT", "/home/cix/vllm-plugin-FL-int8")
).resolve()


def preload_codegen_backend() -> None:
    """Make the unmodified plugin's `kleidiai` selector call codegen enable."""
    from vllm_fl.ops import cpu_int8_kai as kai

    name = "vllm_fl.ops.cpu_int8_triton_codegen"
    source = ROOT / "integrations/vllm/cpu_int8_triton_codegen.py"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # Keep the checkout read-only for benchmarking.  register_model() imports
    # this module attribute after vLLM discovers the existing plugin.
    kai.enable_int8 = module.enable_int8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("codegen", "kleidiai"))
    parser.add_argument(
        "--model", type=Path, default=Path("/home/cix/models/Qwen3-0.6B")
    )
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--profile-triton-time",
        action="store_true",
        help=(
            "time generated C API launches and complete W8 custom ops; "
            "excluded from normal performance runs"
        ),
    )
    parser.add_argument(
        "--p0-codegen",
        action="store_true",
        help="enable audited decode RMSNorm and optional RoPE AOT routes",
    )
    parser.add_argument(
        "--p0-ops",
        default="rms,fused_rms,rope,swiglu",
        help="comma-separated P0 routes used with --p0-codegen",
    )
    parser.add_argument(
        "--p0-bundle",
        type=Path,
        default=(
            Path(os.environ["FL_CPU_P0_TRITON_BUNDLE"])
            if "FL_CPU_P0_TRITON_BUNDLE" in os.environ
            else None
        ),
        help="exact target/codegen-key P0 Norm AOT bundle",
    )
    parser.add_argument(
        "--p0-library",
        type=Path,
        default=Path(
            os.environ.get(
                "FL_CPU_P0_TRITON_LIBRARY",
                ROOT
                / "artifacts/vllm-triton-backend/"
                "libtriton_p0_norm_backend.so",
            )
        ),
        help="minimal P0 Norm AOT launcher",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--prompt", default="What is 2+3? Think step by step."
    )
    parser.add_argument(
        "--expected-token-digest",
        help="fail unless greedy output matches a reference backend run",
    )
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
    args = parser.parse_args()
    if args.tokens < 2 or args.rounds < 1 or args.warmup < 0:
        raise ValueError("tokens>=2, rounds>=1, warmup>=0 required")
    execution_mode = "eager" if args.eager else "compiled"

    sys.path.insert(0, str(PLUGIN))
    os.environ["VLLM_PLUGINS"] = "fl"
    os.environ["FL_CPU_INT8"] = "1"
    # The codegen A/B preloads the replacement enable function without
    # modifying the external plugin checkout.
    os.environ["FL_CPU_INT8_BACKEND"] = "kleidiai"
    os.environ["FL_CPU_INT8_STRICT"] = "1"
    os.environ["FL_INT8_LMHEAD"] = "0"
    os.environ["FL_CPU_UNIPROC"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if args.profile_triton_time:
        os.environ["FL_CPU_INT8_TRITON_PROFILE_TIME"] = "1"
    if args.p0_codegen:
        if args.backend != "codegen":
            raise ValueError("--p0-codegen requires the codegen backend")
        if args.p0_bundle is None:
            raise ValueError(
                "--p0-bundle or FL_CPU_P0_TRITON_BUNDLE is required"
            )
        os.environ["FL_CPU_P0_TRITON"] = "1"
        os.environ["FL_CPU_P0_TRITON_OPS"] = args.p0_ops
        os.environ["FL_CPU_P0_TRITON_BUNDLE"] = str(
            args.p0_bundle.resolve()
        )
        os.environ["FL_CPU_P0_TRITON_LIBRARY"] = str(
            args.p0_library.resolve()
        )
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "2")
    os.environ.setdefault(
        "FL_CPU_INT8_TRITON_LIBRARY",
        str(
            ROOT
            / "artifacts/vllm-triton-backend/"
            "libtriton_kai_w8_decode_backend.so"
        ),
    )
    if args.backend == "codegen":
        if args.bundle is None:
            raise ValueError(
                "--bundle or FL_CPU_INT8_TRITON_BUNDLE is required for codegen"
            )
        os.environ["FL_CPU_INT8_TRITON_BUNDLE"] = str(
            args.bundle.resolve()
        )
        preload_codegen_backend()

    import torch
    from vllm import LLM, SamplingParams

    torch.set_num_threads(args.threads)
    llm = LLM(
        model=str(args.model.resolve()),
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=args.eager,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.15,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=args.seed,
    )

    import vllm.model_executor.layers.utils as layer_utils

    live = getattr(
        layer_utils, "_fl_int8_triton_codegen_enabled", False
    )
    if (args.backend == "codegen") != bool(live):
        raise RuntimeError(
            f"wrong live backend: requested={args.backend}, codegen={live}"
        )

    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.tokens, ignore_eos=True
    )

    codegen_module = (
        sys.modules["vllm_fl.ops.cpu_int8_triton_codegen"]
        if args.backend == "codegen"
        else None
    )

    def run() -> tuple[
        float, float, float, float, int, list[int], dict[str, int]
    ]:
        stats_before = codegen_module.stats() if codegen_module else {}
        start = time.perf_counter()
        request = llm.generate([args.prompt], sampling)[0]
        wall = time.perf_counter() - start
        stats_after = codegen_module.stats() if codegen_module else {}
        stats_delta = {
            key: stats_after[key] - stats_before.get(key, 0)
            for key in stats_after
        }
        token_ids = request.outputs[0].token_ids
        tokens = len(token_ids)
        if tokens != args.tokens:
            raise RuntimeError(
                f"expected {args.tokens} generated tokens, got {tokens}"
            )
        metrics = request.metrics
        if (
            metrics.scheduled_ts is None
            or metrics.first_token_ts is None
            or metrics.last_token_ts is None
        ):
            raise RuntimeError("vLLM request did not report inference timestamps")
        prefill_time = metrics.first_token_ts - metrics.scheduled_ts
        decode_time = metrics.last_token_ts - metrics.first_token_ts
        if prefill_time <= 0 or decode_time <= 0:
            raise RuntimeError(
                f"invalid prefill/decode duration: {prefill_time}/{decode_time}"
            )
        decode_tps = (tokens - 1) / decode_time
        return (
            decode_tps,
            wall,
            prefill_time * 1000.0,
            (metrics.last_token_ts - metrics.scheduled_ts) * 1000.0,
            len(request.prompt_token_ids),
            token_ids,
            stats_delta,
        )

    for _ in range(args.warmup):
        run()
    samples = []
    walls = []
    prefills = []
    inference_times = []
    prompt_sizes = []
    token_sequences = []
    profile_deltas = []
    for index in range(args.rounds):
        (
            decode_tps,
            wall,
            prefill_ms,
            inference_ms,
            prompt_tokens,
            token_ids,
            profile_delta,
        ) = run()
        samples.append(decode_tps)
        walls.append(wall)
        prefills.append(prefill_ms)
        inference_times.append(inference_ms)
        prompt_sizes.append(prompt_tokens)
        token_sequences.append(token_ids)
        profile_deltas.append(profile_delta)
        profile_text = ""
        if args.profile_triton_time and args.backend == "codegen":
            w8_launch_ns = sum(
                profile_delta[f"m{m}_codegen_launch_ns"]
                for m in (1, 4, 8, 12, 16)
            )
            w8_op_ns = sum(
                profile_delta[f"m{m}_codegen_op_ns"]
                for m in (1, 4, 8, 12, 16)
            )
            p0_routes = (
                (
                    "rms_codegen",
                    "fused_rms_codegen",
                    "rope_codegen",
                    "swiglu_codegen",
                )
                if args.p0_codegen
                else ()
            )
            launch_ns = w8_launch_ns + sum(
                profile_delta[f"{route}_launch_ns"] for route in p0_routes
            )
            op_ns = w8_op_ns + sum(
                profile_delta[f"{route}_op_ns"] for route in p0_routes
            )
            profile_text = (
                f" triton_launch_ms={launch_ns / 1e6:.6f}"
                f" triton_op_ms={op_ns / 1e6:.6f}"
                f" triton_launch_pct={launch_ns / 1e4 / inference_ms:.3f}"
                f" triton_op_pct={op_ns / 1e4 / inference_ms:.3f}"
            )
        print(
            f"round={index + 1} prompt_tokens={prompt_tokens} "
            f"tokens={len(token_ids)} prefill_ms={prefill_ms:.6f} "
            f"inference_ms={inference_ms:.6f} "
            f"decode_tps={decode_tps:.6f} wall_s={wall:.6f}"
            f"{profile_text}",
            flush=True,
        )
    if any(size != prompt_sizes[0] for size in prompt_sizes[1:]):
        raise AssertionError("prompt token count changed between measured rounds")
    if any(tokens != token_sequences[0] for tokens in token_sequences[1:]):
        raise AssertionError("greedy token IDs changed between measured rounds")
    token_text = ",".join(str(value) for value in token_sequences[0])
    token_digest = hashlib.sha256(token_text.encode()).hexdigest()
    if (
        args.expected_token_digest is not None
        and token_digest != args.expected_token_digest
    ):
        raise AssertionError(
            "token digest mismatch: "
            f"actual={token_digest}, expected={args.expected_token_digest}"
        )
    print(
        f"PASS vLLM W8 E2E backend={args.backend}\n"
        f"model={args.model.resolve()}\n"
        f"execution_mode={execution_mode}\n"
        f"threads={torch.get_num_threads()}\n"
        f"seed={args.seed}\n"
        f"prompt_tokens={prompt_sizes[0]}\n"
        f"median_prefill_ms={statistics.median(prefills):.6f}\n"
        f"median_inference_ms={statistics.median(inference_times):.6f}\n"
        f"median_decode_tps={statistics.median(samples):.6f}\n"
        f"median_wall_s={statistics.median(walls):.6f}\n"
        f"codegen_live={bool(live)}\n"
        f"token_digest={token_digest}\n"
        f"token_ids={token_text}"
    )
    if args.backend == "codegen":
        codegen_stats = codegen_module.stats()
        if args.threads <= codegen_module.M1_MAX_THREADS:
            assert codegen_stats["m1_codegen_calls"] > 0
        else:
            assert codegen_stats["m1_codegen_calls"] == 0
        print(f"codegen_stats={codegen_stats}")
        if args.profile_triton_time:
            route_profile = {}
            for m in (1, 4, 8, 12, 16):
                prefix = f"m{m}_codegen"
                route_profile[f"m{m}"] = {
                    "calls": sum(
                        delta[f"{prefix}_calls"] for delta in profile_deltas
                    ),
                    "launch_ms": sum(
                        delta[f"{prefix}_launch_ns"]
                        for delta in profile_deltas
                    )
                    / 1e6,
                    "op_ms": sum(
                        delta[f"{prefix}_op_ns"]
                        for delta in profile_deltas
                    )
                    / 1e6,
                }
            if args.p0_codegen:
                for label, prefix in (
                    ("rms", "rms_codegen"),
                    ("fused_rms", "fused_rms_codegen"),
                    ("rope", "rope_codegen"),
                    ("swiglu", "swiglu_codegen"),
                ):
                    route_profile[label] = {
                        "calls": sum(
                            delta[f"{prefix}_calls"]
                            for delta in profile_deltas
                        ),
                        "launch_ms": sum(
                            delta[f"{prefix}_launch_ns"]
                            for delta in profile_deltas
                        )
                        / 1e6,
                        "op_ms": sum(
                            delta[f"{prefix}_op_ns"]
                            for delta in profile_deltas
                        )
                        / 1e6,
                    }
            total_inference_ms = sum(inference_times)
            total_launch_ms = sum(
                value["launch_ms"] for value in route_profile.values()
            )
            total_op_ms = sum(
                value["op_ms"] for value in route_profile.values()
            )
            fallback_calls = sum(
                delta["kai_fallback_calls"] for delta in profile_deltas
            )
            print(
                f"profile_measured_rounds={args.rounds}\n"
                f"profile_inference_ms_total={total_inference_ms:.6f}\n"
                f"profile_triton_launch_ms_total={total_launch_ms:.6f}\n"
                f"profile_triton_op_ms_total={total_op_ms:.6f}\n"
                f"profile_triton_launch_pct="
                f"{100.0 * total_launch_ms / total_inference_ms:.6f}\n"
                f"profile_triton_op_pct="
                f"{100.0 * total_op_ms / total_inference_ms:.6f}\n"
                f"profile_kai_fallback_calls={fallback_calls}\n"
                f"profile_by_route={route_profile}"
            )


if __name__ == "__main__":
    main()
