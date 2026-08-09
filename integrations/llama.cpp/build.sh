#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
integration_dir="${repo_dir}/integrations/llama.cpp"
jit_dir="${repo_dir}/third_party/libtriton_jit"
out_dir="${repo_dir}/artifacts/llama-triton-backend"
mkdir -p "${out_dir}"

g++ -std=c++20 -O3 -DNDEBUG -fPIC -shared \
  -I"${integration_dir}" -I"${jit_dir}/include" \
  "${integration_dir}/triton_w8_backend.cpp" \
  "${jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${out_dir}/libtriton_w8_backend.so"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${integration_dir}" \
  "${integration_dir}/test_backend.cpp" \
  -L"${out_dir}" -ltriton_w8_backend -ldl \
  -Wl,-rpath,'$ORIGIN' \
  -o "${out_dir}/test_triton_w8_backend"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${integration_dir}" \
  "${integration_dir}/bench_backend_threadpool.cpp" \
  -L"${out_dir}" -ltriton_w8_backend -ldl -pthread \
  -Wl,-rpath,'$ORIGIN' \
  -o "${out_dir}/bench_triton_w8_threadpool"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${integration_dir}" \
  "${integration_dir}/bench_backend_split_threadpool.cpp" \
  -L"${out_dir}" -ltriton_w8_backend -ldl -pthread \
  -Wl,-rpath,'$ORIGIN' \
  -o "${out_dir}/bench_triton_w8_split_threadpool"

printf '%s\n' \
  "${out_dir}/libtriton_w8_backend.so" \
  "${out_dir}/test_triton_w8_backend" \
  "${out_dir}/bench_triton_w8_threadpool" \
  "${out_dir}/bench_triton_w8_split_threadpool"

g++ -std=c++20 -O3 -DNDEBUG -fPIC -shared \
  -I"${integration_dir}" \
  "${integration_dir}/triton_w4_backend.cpp" \
  -ldl \
  -o "${out_dir}/libtriton_w4_backend.so"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${integration_dir}" \
  "${integration_dir}/test_w4_backend.cpp" \
  -L"${out_dir}" -ltriton_w4_backend -ldl \
  -Wl,-rpath,'$ORIGIN' \
  -o "${out_dir}/test_triton_w4_backend"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${integration_dir}" \
  "${integration_dir}/test_w4_q41_backend.cpp" \
  -L"${out_dir}" -ltriton_w4_backend -ldl \
  -Wl,-rpath,'$ORIGIN' \
  -o "${out_dir}/test_triton_w4_q41_backend"

printf '%s\n' \
  "${out_dir}/libtriton_w4_backend.so" \
  "${out_dir}/test_triton_w4_backend" \
  "${out_dir}/test_triton_w4_q41_backend"
