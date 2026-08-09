#!/usr/bin/env python3
"""MiniCPM5 vLLM ARM Q4/Q8 A/B: Triton versus native vLLM kernels.

For a BF16 checkpoint, the plugin routes compare the same signed Q4 block-32
weights.  For a compressed-tensors W4A8-G128 checkpoint, ``vllm_native`` uses
vLLM's ATen/KleidiAI Dynamic4bitLinearKernel and ``libtriton_jit`` consumes the
same checkpoint INT4 values/scales through ordinary Triton SDOT/I8MM kernels.
Weight packing and model construction are outside measured request time.
For a compressed-tensors W8A8 checkpoint, ``vllm_native`` uses oneDNN while
``libtriton_jit`` routes eligible dynamic per-token W8 linears through ordinary
Triton SDOT/I8MM kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("FLAGGEMS_VENDOR_NAME", "arm")
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: int(value) - int(before.get(key, 0)) for key, value in after.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend",
        choices=(
            "triton_codegen",
            "libtriton_jit",
            "tleraw",
            "vllm_native",
            "bf16_native",
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/cix/"
            "MiniCPM5-2.6B-0426_job_327123_step_24000_fusion_think_512k"
        ),
    )
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=12)
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help="comma-separated token IDs appended to the deterministic prompt",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=0,
        help="return this many top log probabilities per generated token",
    )
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="vLLM compilation mode; the deployment guide uses mode 3",
    )
    parser.add_argument(
        "--legacy-q8-int4-selector",
        action="store_true",
        help=(
            "compatibility only: select the unified operator through an old "
            "plugin's INT4 backend; patched production plugins use INT8"
        ),
    )
    parser.add_argument(
        "--lm-head",
        choices=("auto", "native", "q4-g32", "w8-channel"),
        default="auto",
        help=(
            "lm_head format; auto follows the deployment contract: G32 for "
            "a Q4 checkpoint and per-channel W8 for a Q8 checkpoint"
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="profile one extra request and write CPU operator aggregates",
    )
    parser.add_argument("--expected-token-digest")
    parser.add_argument(
        "--libtriton-jit-op",
        type=Path,
        default=ROOT
        / "artifacts/vllm-libtriton-jit-q4/build/libtriton_jit_q4_op.so",
    )
    args = parser.parse_args()
    if args.tokens < 2 or args.rounds < 1 or args.warmup < 0:
        raise ValueError("tokens>=2, rounds>=1, warmup>=0 required")
    if args.prompt_tokens < 1:
        raise ValueError("prompt-tokens must be positive")

    model = args.model.resolve()
    config_path = model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    model_config = json.loads(config_path.read_text())
    model_quantization = model_config.get("quantization_config")
    checkpoint_is_quantized = model_quantization is not None
    quant_bits = None
    if model_quantization is not None:
        config_groups = model_quantization.get("config_groups", {})
        if config_groups:
            first_group = next(iter(config_groups.values()))
            quant_bits = first_group.get("weights", {}).get("num_bits")
    lm_head_format = args.lm_head
    if lm_head_format == "auto":
        if quant_bits == 8:
            lm_head_format = "w8-channel"
        elif quant_bits == 4 or not checkpoint_is_quantized:
            lm_head_format = "q4-g32"
        else:
            lm_head_format = "native"

    os.environ["VLLM_PLUGINS"] = "fl"
    os.environ["FL_CPU_UNIPROC"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["FL_CPU_OMP_ACTIVE_WAIT"] = "1"
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
    os.environ["FL_INT4_LMHEAD"] = "1" if lm_head_format == "q4-g32" else "0"
    os.environ["FL_INT8_LMHEAD"] = (
        "1" if lm_head_format == "w8-channel" else "0"
    )
    os.environ["FL_CPU_INT4_STRICT"] = "1"
    os.environ["FL_CPU_INT8_STRICT"] = "1"
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "2")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    if args.backend in {"bf16_native", "vllm_native"}:
        os.environ["FL_CPU_INT4"] = "0"
        os.environ["FL_CPU_INT8"] = "0"
    else:
        q8_selector = quant_bits == 8 and not args.legacy_q8_int4_selector
        if q8_selector and args.backend != "libtriton_jit":
            raise ValueError(
                "the Q8 production selector currently supports "
                "backend=libtriton_jit in this benchmark"
            )
        os.environ["FL_CPU_INT4"] = "0" if q8_selector else "1"
        os.environ["FL_CPU_INT8"] = "1" if q8_selector else "0"
        selector = "FL_CPU_INT8_BACKEND" if q8_selector else "FL_CPU_INT4_BACKEND"
        os.environ[selector] = args.backend
        if args.backend == "libtriton_jit":
            library = args.libtriton_jit_op.resolve()
            if not library.is_file():
                raise FileNotFoundError(f"libtriton_jit Q4 op not found: {library}")
            os.environ["FLAGGEMS_LIBTRITON_JIT_Q4_OP"] = str(library)

    import torch
    from vllm import LLM, SamplingParams

    torch.set_num_threads(args.threads)
    llm = LLM(
        model=str(model),
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=not args.compiled,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.15,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=args.seed,
        compilation_config={"mode": args.compile_mode if args.compiled else 0},
    )
    engine_config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
    effective_compile_mode = 0
    if engine_config is not None:
        effective_compile_mode = int(engine_config.compilation_config.mode)

    from vllm.model_executor.layers import utils as layer_utils

    codegen_live = bool(
        getattr(layer_utils, "_flag_gems_q4_codegen_enabled", False)
    )
    codegen_runtime = getattr(
        layer_utils, "_flag_gems_q4_codegen_runtime", None
    )
    tleraw_live = bool(getattr(layer_utils, "_fl_tleraw_int4_enabled", False))
    expected_live = {
        "triton_codegen": (True, "python", False),
        "libtriton_jit": (True, "libtriton_jit", False),
        "tleraw": (False, None, True),
        "vllm_native": (False, None, False),
        "bf16_native": (False, None, False),
    }[args.backend]
    if (codegen_live, codegen_runtime, tleraw_live) != expected_live:
        raise RuntimeError(
            "wrong live backend: "
            f"requested={args.backend}, codegen={codegen_live}, "
            f"runtime={codegen_runtime}, tleraw={tleraw_live}"
        )

    if args.backend in {"triton_codegen", "libtriton_jit"}:
        from flag_gems.runtime.backend._arm.q4 import stats as backend_stats
    elif args.backend == "tleraw":
        from vllm_fl.ops.cpu_int4_tleraw import stats as backend_stats
    else:
        backend_stats = lambda: {}

    # Token IDs avoid tokenizer and chat-template differences.  They are also
    # stable across all three routes and contain no special tokens.
    prompt = [1000 + ((index * 7919 + args.seed) % 100000)
              for index in range(args.prompt_tokens)]
    if args.prompt_suffix:
        prompt.extend(int(value) for value in args.prompt_suffix.split(","))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.tokens,
        ignore_eos=True,
        logprobs=args.logprobs or None,
    )

    def run() -> dict[str, object]:
        before = backend_stats()
        start = time.perf_counter()
        request = llm.generate({"prompt_token_ids": prompt}, sampling)[0]
        wall_s = time.perf_counter() - start
        after = backend_stats()
        token_ids = list(request.outputs[0].token_ids)
        if len(token_ids) != args.tokens:
            raise RuntimeError(
                f"expected {args.tokens} generated tokens, got {len(token_ids)}"
            )
        metrics = request.metrics
        if (
            metrics.scheduled_ts is None
            or metrics.first_token_ts is None
            or metrics.last_token_ts is None
        ):
            raise RuntimeError("vLLM did not report inference timestamps")
        prefill_s = metrics.first_token_ts - metrics.scheduled_ts
        decode_s = metrics.last_token_ts - metrics.first_token_ts
        if prefill_s <= 0 or decode_s <= 0:
            raise RuntimeError(
                f"invalid prefill/decode time: {prefill_s}/{decode_s}"
            )
        token_logprobs = None
        if request.outputs[0].logprobs is not None:
            token_logprobs = [
                {
                    str(token_id): {
                        "logprob": float(value.logprob),
                        "rank": value.rank,
                    }
                    for token_id, value in position.items()
                }
                for position in request.outputs[0].logprobs
            ]
        return {
            "prefill_ms": prefill_s * 1000.0,
            "decode_ms": decode_s * 1000.0,
            "inference_ms": (metrics.last_token_ts - metrics.scheduled_ts)
            * 1000.0,
            "decode_tps": (len(token_ids) - 1) / decode_s,
            "wall_s": wall_s,
            "token_ids": token_ids,
            "token_logprobs": token_logprobs,
            "stats_delta": _delta(after, before),
        }

    for index in range(args.warmup):
        sample = run()
        print(
            f"warmup={index + 1} inference_ms={sample['inference_ms']:.3f}",
            flush=True,
        )

    samples = []
    for index in range(args.rounds):
        sample = run()
        samples.append(sample)
        print(
            f"round={index + 1} prefill_ms={sample['prefill_ms']:.3f} "
            f"decode_ms={sample['decode_ms']:.3f} "
            f"decode_tps={sample['decode_tps']:.3f} "
            f"inference_ms={sample['inference_ms']:.3f} "
            f"stats_delta={sample['stats_delta']}",
            flush=True,
        )

    reference_tokens = samples[0]["token_ids"]
    if any(sample["token_ids"] != reference_tokens for sample in samples[1:]):
        raise AssertionError("greedy token IDs changed between measured rounds")
    token_text = ",".join(str(value) for value in reference_tokens)
    token_digest = hashlib.sha256(token_text.encode()).hexdigest()
    if (
        args.expected_token_digest is not None
        and token_digest != args.expected_token_digest
    ):
        raise AssertionError(
            f"token digest mismatch: {token_digest} != "
            f"{args.expected_token_digest}"
        )

    def median(key: str) -> float:
        return float(statistics.median(float(sample[key]) for sample in samples))

    result = {
        "backend": args.backend,
        "model": str(model),
        "checkpoint_quantized": checkpoint_is_quantized,
        "checkpoint_weight_bits": quant_bits,
        "model_quantization": model_quantization,
        "lm_head_format": lm_head_format,
        "execution_mode": "compiled" if args.compiled else "eager",
        "compile_mode": args.compile_mode if args.compiled else 0,
        "effective_compile_mode": effective_compile_mode,
        "threads": torch.get_num_threads(),
        "prompt_tokens": args.prompt_tokens,
        "prompt_suffix": args.prompt_suffix,
        "effective_prompt_tokens": len(prompt),
        "output_tokens": args.tokens,
        "requested_logprobs": args.logprobs,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "median_prefill_ms": median("prefill_ms"),
        "median_decode_ms": median("decode_ms"),
        "median_decode_tps": median("decode_tps"),
        "median_inference_ms": median("inference_ms"),
        "median_wall_s": median("wall_s"),
        "token_digest": token_digest,
        "token_ids": reference_tokens,
        "backend_stats": backend_stats(),
        "samples": samples,
    }
    if args.profile_json is not None:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU]) as profiler:
            profiled_sample = run()
        events = []
        for event in profiler.key_averages():
            events.append(
                {
                    "name": event.key,
                    "calls": int(event.count),
                    "self_cpu_us": float(event.self_cpu_time_total),
                    "cpu_total_us": float(event.cpu_time_total),
                }
            )
        events.sort(key=lambda item: item["self_cpu_us"], reverse=True)
        profile_result = {
            "backend": args.backend,
            "execution_mode": result["execution_mode"],
            "inference_ms": profiled_sample["inference_ms"],
            "decode_tps": profiled_sample["decode_tps"],
            "events": events,
        }
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps(profile_result, indent=2) + "\n")
        print(
            "PROFILE_TOP="
            + json.dumps(profile_result["events"][:20], sort_keys=True),
            flush=True,
        )
        print(f"profile_wrote={args.profile_json}", flush=True)
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote={args.json_out}", flush=True)


if __name__ == "__main__":
    main()
