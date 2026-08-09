#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "${script_dir}/build_qwen3_p0_norm_codegen_bundle.sh" "$@"
