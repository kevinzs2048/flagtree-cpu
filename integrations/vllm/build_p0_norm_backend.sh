#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
source_dir="${repo_dir}/integrations/vllm"
out_dir="${repo_dir}/artifacts/vllm-triton-backend"
mkdir -p "${out_dir}"

"${CXX:-g++}" -std=c++17 -O3 -DNDEBUG -fPIC -shared \
  -Wall -Wextra -Werror \
  -I"${source_dir}" \
  "${source_dir}/triton_p0_norm_backend.cpp" \
  -ldl \
  -o "${out_dir}/libtriton_p0_norm_backend.so"

printf '%s\n' "${out_dir}/libtriton_p0_norm_backend.so"
