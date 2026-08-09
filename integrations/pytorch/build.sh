#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
integration_dir="${repo_dir}/integrations/pytorch"
jit_dir="${repo_dir}/third_party/libtriton_jit"
out_dir="${repo_dir}/artifacts/pytorch-triton-backend"
mkdir -p "${out_dir}"

g++ -std=c++20 -O3 -DNDEBUG -fPIC -shared \
  -I"${integration_dir}" -I"${jit_dir}/include" \
  "${integration_dir}/triton_bf16_w8_backend.cpp" \
  -ldl -pthread \
  -o "${out_dir}/libtriton_bf16_w8_backend.so"

printf '%s\n' "${out_dir}/libtriton_bf16_w8_backend.so"
