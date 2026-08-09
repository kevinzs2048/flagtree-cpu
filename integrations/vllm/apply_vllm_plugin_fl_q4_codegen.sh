#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/vllm-plugin-FL" >&2
  exit 2
fi

plugin_dir="$(cd "$1" && pwd)"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
patch_files=(
  "${repo_dir}/integrations/vllm/vllm_plugin_fl_resolved_quant_codegen.patch"
  "${repo_dir}/integrations/vllm/vllm_plugin_fl_current_quant_codegen.patch"
  "${repo_dir}/integrations/vllm/vllm_plugin_fl_q4_codegen.patch"
)
patch_file=""
for candidate in "${patch_files[@]}"; do
  if git -C "${plugin_dir}" apply --check "${candidate}" 2>/dev/null; then
    patch_file="${candidate}"
    break
  fi
done
if [[ -z "${patch_file}" ]]; then
  echo "No supported vllm-plugin-FL layout matched" >&2
  exit 1
fi

git -C "${plugin_dir}" apply "${patch_file}"
echo "Applied Q4/Q8 Triton-codegen backend options to ${plugin_dir}"
echo "Patch variant: $(basename "${patch_file}")"
echo "Select JIT C++ routing with FL_CPU_INT4_BACKEND=libtriton_jit"
echo "or FL_CPU_INT8_BACKEND=libtriton_jit"
