#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
output_dir="${repo_dir}/artifacts/vllm-triton-backend"
mkdir -p "${output_dir}"

parallel_flags=()
link_flags=()
system="$(uname -s)"
if [[ -n "${CXX:-}" ]]; then
  cxx="${CXX}"
elif [[ "${system}" == "Darwin" ]] && command -v clang++ >/dev/null 2>&1; then
  cxx="$(command -v clang++)"
elif command -v g++ >/dev/null 2>&1; then
  cxx="$(command -v g++)"
elif command -v clang++ >/dev/null 2>&1; then
  cxx="$(command -v clang++)"
else
  printf 'no C++ compiler found; set CXX explicitly\n' >&2
  exit 1
fi
if [[ "${system}" == "Linux" ]]; then
  parallel_flags=(-fopenmp)
  link_flags=(-ldl)
elif [[ "${system}" != "Darwin" ]]; then
  printf 'unsupported host OS: %s\n' "${system}" >&2
  exit 1
fi

"${cxx}" -std=c++20 -O3 -DNDEBUG -fPIC -shared \
  "${parallel_flags[@]}" \
  "${repo_dir}/integrations/vllm/triton_kai_w8_decode_backend.cpp" \
  "${link_flags[@]}" \
  -o "${output_dir}/libtriton_kai_w8_decode_backend.so"

printf '%s\n' \
  "compiler=${cxx}" \
  "system=${system}" \
  "${output_dir}/libtriton_kai_w8_decode_backend.so"
