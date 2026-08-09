#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
VLLM_ROOT=${VLLM_ROOT:-/Users/kevin/qwen36-w4a8/vllm-0.19.0}
MODEL_DIR=${MODEL_DIR:-/Users/kevin/models/Qwen3.5-9B-W4A8-G32}
TRITON_ROOT="$REPO_ROOT/ports/triton-cpu-3.7.2"
FLAGGEMS_ROOT="$REPO_ROOT/third_party/FlagGems"
Q4_OP="$REPO_ROOT/artifacts/vllm-libtriton-jit-q4/build/libtriton_jit_q4_op.dylib"

if [ ! -x "$VLLM_ROOT/.venv/bin/python" ]; then
  echo "vLLM virtualenv not found: $VLLM_ROOT/.venv" >&2
  exit 2
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "model not found: $MODEL_DIR" >&2
  exit 2
fi
if [ ! -f "$Q4_OP" ]; then
  echo "Q4 operator not found: $Q4_OP" >&2
  exit 2
fi

ulimit -n 4096
export PYTHONPATH="$TRITON_ROOT/python:$FLAGGEMS_ROOT/src:$VLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=triton_cpu_qwen35
export FLAGGEMS_VENDOR_NAME=arm
export FLAGGEMS_LIBTRITON_JIT_Q4_OP="$Q4_OP"
export TRITON_LOCAL_LIBOMP_PATH=${TRITON_LOCAL_LIBOMP_PATH:-/opt/homebrew/opt/libomp}
export VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-2}
export VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "$VLLM_ROOT/.venv/bin/python" \
  "$SCRIPT_DIR/run_qwen35_m4_w4a8.py" "$MODEL_DIR" "$@"
