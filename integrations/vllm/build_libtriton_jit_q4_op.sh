#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
source_dir="${repo_dir}/integrations/vllm/libtriton_jit_q4"
libtriton_jit_root="${repo_dir}/third_party/libtriton_jit"
libtriton_jit_build="${libtriton_jit_root}/build-vllm"
build_dir="${repo_dir}/artifacts/vllm-libtriton-jit-q4/build"
python_bin="${VLLM_PYTHON:-$(command -v python3)}"
python_root="${VLLM_PYTHON_ROOT:-$("${python_bin}" -c 'import sys; print(sys.prefix)')}"
cmake_bin="${CMAKE_BIN:-$(command -v cmake)}"
ninja_bin="${NINJA_BIN:-$(command -v ninja)}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  cxx_bin="${CXX:-$(command -v clang++)}"
  if [[ -d /opt/homebrew/opt/libffi/lib/pkgconfig ]]; then
    export PKG_CONFIG_PATH="/opt/homebrew/opt/libffi/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  fi
  openmp_args=()
  if [[ -d /opt/homebrew/opt/libomp ]]; then
    openmp_args+=("-DOpenMP_ROOT=/opt/homebrew/opt/libomp")
  fi
  library_suffix="dylib"
else
  cxx_bin="${CXX:-$(command -v c++)}"
  openmp_args=()
  library_suffix="so"
fi

export VIRTUAL_ENV="${python_root}"

# A clean recursive clone has no preconfigured libtriton_jit build. Configure
# the CPU backend against the same Python/PyTorch ABI as the vLLM operator.
"${cmake_bin}" -S "${libtriton_jit_root}" -B "${libtriton_jit_build}" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="${ninja_bin}" \
  -DCMAKE_CXX_COMPILER="${cxx_bin}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_ROOT="${python_root}" \
  -DBACKEND=CPU \
  -DTRITON_JIT_BUILD_OPERATORS=OFF \
  "${openmp_args[@]}"
"${cmake_bin}" --build "${libtriton_jit_build}" --parallel 4

"${cmake_bin}" -S "${source_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="${ninja_bin}" \
  -DCMAKE_CXX_COMPILER="${cxx_bin}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_ROOT="${python_root}" \
  -DTRITON_JIT_ROOT="${libtriton_jit_root}" \
  -DTRITON_JIT_BUILD="${libtriton_jit_build}" \
  "${openmp_args[@]}"
"${cmake_bin}" --build "${build_dir}" -j4

# The Q4 operator links libtriton_jit through a loader-relative rpath.  Keep
# both libraries in one artifact directory so it can be moved as a bundle.
"${cmake_bin}" -E copy_if_different \
  "${libtriton_jit_build}/src/libtriton_jit.${library_suffix}" \
  "${build_dir}/libtriton_jit.${library_suffix}"

printf '%s\n' "${build_dir}/libtriton_jit_q4_op.${library_suffix}"
