#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/vllm-plugin-FL-int8" >&2
  exit 2
fi

plugin_dir="$(cd "$1" && pwd)"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
patch_file="${repo_dir}/integrations/vllm/vllm_plugin_fl_resolved_quant_codegen.patch"

git -C "${plugin_dir}" apply --check "${patch_file}"
git -C "${plugin_dir}" apply "${patch_file}"
echo "Applied resolved Q4/Q8 Triton-codegen backends to ${plugin_dir}"
