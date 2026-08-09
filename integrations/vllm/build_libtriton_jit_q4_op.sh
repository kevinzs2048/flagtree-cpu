#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
source_dir="${repo_dir}/integrations/vllm/libtriton_jit_q4"
libtriton_jit_root="${repo_dir}/third_party/libtriton_jit"
libtriton_jit_build="${libtriton_jit_root}/build-vllm"
build_dir="${repo_dir}/artifacts/vllm-libtriton-jit-q4/build"
python_root="${VLLM_PYTHON_ROOT:-/home/kevin/venv-fl-020}"
cmake_bin="${CMAKE_BIN:-/home/cix/venv-fep-e2e/bin/cmake}"
ninja_bin="${NINJA_BIN:-/home/cix/venv-fep-e2e/bin/ninja}"

"${cmake_bin}" -S "${source_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="${ninja_bin}" \
  -DCMAKE_CXX_COMPILER="${CXX:-/usr/bin/c++}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_ROOT="${python_root}" \
  -DTRITON_JIT_ROOT="${libtriton_jit_root}" \
  -DTRITON_JIT_BUILD="${libtriton_jit_build}"
"${cmake_bin}" --build "${build_dir}" -j4

printf '%s\n' "${build_dir}/libtriton_jit_q4_op.so"
