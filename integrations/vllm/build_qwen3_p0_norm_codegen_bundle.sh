#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
port_dir="${repo_dir}/ports/triton-cpu-3.7.2"
python_bin="${TRITON_TEST_PYTHON:-/home/kevin/venv-int8-clean/bin/python}"
venv_site="${TRITON_CPU_VENV_SITE:-/home/kevin/venv-int8-clean/lib/python3.11/site-packages}"
dependency_site="${TRITON_CPU_DEPENDENCY_SITE:-/home/cix/venv-fep-e2e/lib/python3.11/site-packages}"
python_path="${port_dir}/python:${repo_dir}/third_party/FlagGems/src:${venv_site}:${dependency_site}"
bundle_root="${repo_dir}/artifacts/vllm-triton-backend/qwen3-p0-norm-codegen"
mkdir -p "${bundle_root}"

target_record="$(
  env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
    TRITON_CPU_BACKEND=1 "${python_bin}" -S - <<'PY'
import platform

from triton.compiler.compiler import make_backend
from triton.runtime.driver import driver

backend = make_backend(driver.active.get_current_target())
mode = "sve2_i8mm" if backend.use_sve2_i8mm() else "fixed"
required = {"bf16"}
if mode == "sve2_i8mm":
    required.add("sve")
print("|".join((
    platform.system().lower(), backend.cpu_arch, backend.cpu_name, mode,
    str(backend.sve_vector_bits), ",".join(sorted(required)), backend.hash(),
)))
PY
)"
IFS='|' read -r target_os target_arch target_cpu target_mode \
  sve_vector_bits required_features backend_hash <<<"${target_record}"
target_tag="${target_os}-${target_arch}-${target_mode}-vl${sve_vector_bits}"
target_tag="$(printf '%s' "${target_tag}" | tr -c 'A-Za-z0-9._-' '_')"

hash_files() {
  TARGET_RECORD="${target_record}" "${python_bin}" -S - "$@" <<'PY'
import hashlib
import os
import pathlib
import sys

digest = hashlib.sha256(os.environ["TARGET_RECORD"].encode())
for value in sys.argv[1:]:
    path = pathlib.Path(value)
    digest.update(str(path).encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
}

compile_key="$(hash_files \
  "${repo_dir}/benchmarks/bench_rmsnorm_ordinary_aot.py" \
  "${repo_dir}/benchmarks/bench_fused_add_rms_ordinary_aot.py" \
  "${repo_dir}/benchmarks/bench_rope_ordinary_aot.py" \
  "${repo_dir}/benchmarks/bench_bf16_w8_mlp_three_stage.py" \
  "${repo_dir}/third_party/FlagGems/src/flag_gems/runtime/backend/_arm/ops/silu_and_mul.py" \
  "${port_dir}/third_party/cpu/backend/compiler.py" \
  "${port_dir}/third_party/cpu/backend/target_info.py" \
  "${port_dir}/python/triton/runtime/build.py" \
  "${port_dir}/python/triton/_C/libtriton.so")"
codegen_key="$(hash_files \
  "${repo_dir}/integrations/vllm/build_qwen3_p0_norm_codegen_bundle.sh" \
  "${repo_dir}/integrations/vllm/triton_p0_norm_backend.cpp" \
  "${repo_dir}/integrations/vllm/triton_p0_norm_backend.h")"
cache_root="${repo_dir}/artifacts/aot-vllm-p0-norm-3.7-${target_tag}-${compile_key}"
release_parent="${bundle_root}/${target_tag}"
release_dir="${release_parent}/${codegen_key}"
mkdir -p "${release_parent}"

compile_rms() {
  local rows="$1"
  local cols="$2"
  local cache_dir="${cache_root}/rms-m${rows}-n${cols}"
  local object
  object="$(find "${cache_dir}" -type f -name '_rms_norm_aot_kernel.so' \
    -print -quit 2>/dev/null || true)"
  if [[ -z "${object}" ]]; then
    env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
      TRITON_CPU_BACKEND=1 TRITON_CACHE_DIR="${cache_dir}" \
      OMP_NUM_THREADS=1 "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_rmsnorm_ordinary_aot.py" \
      --m "${rows}" --n "${cols}" --warmup 1 --iters 1 --batches 1 \
      >&2 || return
    object="$(find "${cache_dir}" -type f -name '_rms_norm_aot_kernel.so' \
      -print -quit)"
  fi
  printf '%s\n' "${object}"
}

compile_fused_rms() {
  local cache_dir="${cache_root}/vllm-fused-rms-m1-n1024"
  local object
  object="$(find "${cache_dir}" -type f \
    -name '_vllm_fused_add_rms_aot_kernel.so' -print -quit \
    2>/dev/null || true)"
  if [[ -z "${object}" ]]; then
    env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
      TRITON_CPU_BACKEND=1 TRITON_CACHE_DIR="${cache_dir}" \
      OMP_NUM_THREADS=1 "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_fused_add_rms_ordinary_aot.py" \
      --m 1 --n 1024 --vllm-semantics \
      --warmup 1 --iters 1 --batches 1 >&2 || return
    object="$(find "${cache_dir}" -type f \
      -name '_vllm_fused_add_rms_aot_kernel.so' -print -quit)"
  fi
  printf '%s\n' "${object}"
}

compile_rope() {
  local cache_dir="${cache_root}/rope-hq16-hkv8-d128"
  local object
  object="$(find "${cache_dir}" -type f -name '_rope_qk_aot_kernel.so' \
    -print -quit 2>/dev/null || true)"
  if [[ -z "${object}" ]]; then
    env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
      TRITON_CPU_BACKEND=1 TRITON_CACHE_DIR="${cache_dir}" \
      OMP_NUM_THREADS=1 "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_rope_ordinary_aot.py" \
      --q-heads 16 --kv-heads 8 --head-dim 128 \
      --warmup 1 --iters 1 --batches 1 >&2 || return
    object="$(find "${cache_dir}" -type f -name '_rope_qk_aot_kernel.so' \
      -print -quit)"
  fi
  printf '%s\n' "${object}"
}

compile_swiglu() {
  local cache_dir="${cache_root}/swiglu-n3072"
  local object
  object="$(find "${cache_dir}" -type f \
    -name '_bf16_swiglu_inline_exp_kernel.so' -print -quit \
    2>/dev/null || true)"
  if [[ -z "${object}" ]]; then
    env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
      TRITON_CPU_BACKEND=1 TRITON_CACHE_DIR="${cache_dir}" \
      OMP_NUM_THREADS=1 "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_bf16_w8_mlp_three_stage.py" \
      --k 1024 --n 3072 --warmup 1 --iters 1 --batches 1 >&2 || return
    object="$(find "${cache_dir}" -type f \
      -name '_bf16_swiglu_inline_exp_kernel.so' -print -quit)"
  fi
  printf '%s\n' "${object}"
}

audit_object() {
  local object="$1"
  local symbol="$2"
  local stack_policy="${3:-none}"
  if ! nm -D --defined-only "${object}" | grep -q " ${symbol}$"; then
    printf 'missing generated symbol %s in %s\n' "${symbol}" "${object}" >&2
    return 1
  fi
  local body
  body="$(objdump -d --disassemble="${symbol}" "${object}")"
  if grep -Eq '^[[:space:]]+[0-9a-f]+:.*\bbl\b' <<<"${body}"; then
    printf 'external call in generated norm function %s\n' "${object}" >&2
    return 1
  fi
  local stack_lines
  stack_lines="$(grep '\[sp' <<<"${body}" || true)"
  if [[ -n "${stack_lines}" ]]; then
    if [[ "${stack_policy}" != "callee_save" ]] || \
        grep -Ev '\b(str|ldr)[[:space:]]+d[0-9]+|\b(stp|ldp)[[:space:]]+d[0-9]+,[[:space:]]*d[0-9]+' \
          <<<"${stack_lines}" | grep -q .; then
      printf 'non-callee-save stack access in generated function %s\n' \
        "${object}" >&2
      return 1
    fi
  fi
}

rms_m1="$(compile_rms 1 1024)"
rms_m16="$(compile_rms 16 128)"
rms_m8="$(compile_rms 8 128)"
fused_m1="$(compile_fused_rms)"
rope="$(compile_rope)"
swiglu="$(compile_swiglu)"
audit_object "${rms_m1}" _rms_norm_aot_kernel
audit_object "${rms_m16}" _rms_norm_aot_kernel
audit_object "${rms_m8}" _rms_norm_aot_kernel
audit_object "${fused_m1}" _vllm_fused_add_rms_aot_kernel
audit_object "${rope}" _rope_qk_aot_kernel
audit_object "${swiglu}" _bf16_swiglu_inline_exp_kernel callee_save

validate_release() {
  "${python_bin}" -S - "$1" "${target_os}" "${target_arch}" \
    "${required_features}" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
metadata = {}
for line in (root / "bundle.meta").read_text(encoding="utf-8").splitlines():
    if line and not line.startswith("#"):
        key, sep, value = line.partition("=")
        if not sep or key in metadata:
            raise ValueError("invalid bundle metadata")
        metadata[key] = value
expected_meta = {
    "format_version": "3", "backend_abi": "2", "target_os": sys.argv[2],
    "target_arch": sys.argv[3], "required_features": sys.argv[4],
    "objects": "6",
}
for key, value in expected_meta.items():
    if metadata.get(key) != value:
        raise ValueError(f"metadata mismatch: {key}")
lines = (root / "manifest.tsv").read_text(encoding="utf-8").splitlines()
header = ["#kind", "rows", "cols", "shape", "symbol", "sha256"]
if not lines or lines[0].split("\t") != header or len(lines) != 7:
    raise ValueError("invalid manifest")
for line in lines[1:]:
    row = dict(zip(header, line.split("\t")))
    path = root / row["shape"] / f"{row['symbol']}.so"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"non-regular object: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"digest mismatch: {path}")
PY
}

if [[ -f "${release_dir}/bundle.meta" ]]; then
  validate_release "${release_dir}"
  printf '%s\n' "bundle=${release_dir}" "reused=true"
  exit 0
fi

stage="$(mktemp -d "${release_parent}/.p0-stage-${codegen_key}.XXXXXX")"
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT

install_object() {
  local source="$1"
  local shape="$2"
  local symbol="$3"
  mkdir -p "${stage}/${shape}"
  install -m 0444 "${source}" "${stage}/${shape}/${symbol}.so"
}
install_object "${rms_m1}" rms-m1-n1024-e1e-6 _rms_norm_aot_kernel
install_object "${rms_m16}" rms-m16-n128-e1e-6 _rms_norm_aot_kernel
install_object "${rms_m8}" rms-m8-n128-e1e-6 _rms_norm_aot_kernel
install_object "${fused_m1}" vllm-fused-rms-m1-n1024-e1e-6 \
  _vllm_fused_add_rms_aot_kernel
install_object "${rope}" rope-hq16-hkv8-d128 _rope_qk_aot_kernel
install_object "${swiglu}" swiglu-n3072 _bf16_swiglu_inline_exp_kernel

env STAGE="${stage}" TARGET_RECORD="${target_record}" \
  TARGET_OS="${target_os}" TARGET_ARCH="${target_arch}" \
  TARGET_CPU="${target_cpu}" TARGET_MODE="${target_mode}" \
  SVE_BITS="${sve_vector_bits}" REQUIRED_FEATURES="${required_features}" \
  BACKEND_HASH="${backend_hash}" COMPILE_KEY="${compile_key}" \
  CODEGEN_KEY="${codegen_key}" "${python_bin}" -S - <<'PY'
import hashlib
import os
import pathlib

root = pathlib.Path(os.environ["STAGE"])
rows = (
    ("rms", 1, 1024, "rms-m1-n1024-e1e-6", "_rms_norm_aot_kernel"),
    ("rms", 16, 128, "rms-m16-n128-e1e-6", "_rms_norm_aot_kernel"),
    ("rms", 8, 128, "rms-m8-n128-e1e-6", "_rms_norm_aot_kernel"),
    ("fused_rms", 1, 1024, "vllm-fused-rms-m1-n1024-e1e-6",
     "_vllm_fused_add_rms_aot_kernel"),
    ("rope", 24, 128, "rope-hq16-hkv8-d128", "_rope_qk_aot_kernel"),
    ("swiglu", 1, 3072, "swiglu-n3072",
     "_bf16_swiglu_inline_exp_kernel"),
)
manifest = ["#kind\trows\tcols\tshape\tsymbol\tsha256"]
for kind, m, n, shape, symbol in rows:
    path = root / shape / f"{symbol}.so"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.append(f"{kind}\t{m}\t{n}\t{shape}\t{symbol}\t{digest}")
(root / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
keys = (
    "TARGET_OS", "TARGET_ARCH", "TARGET_CPU", "TARGET_MODE", "SVE_BITS",
    "REQUIRED_FEATURES", "BACKEND_HASH", "COMPILE_KEY", "CODEGEN_KEY",
)
metadata = ["format_version=3", "backend_abi=2", "objects=6"]
metadata.extend(f"{key.lower()}={os.environ[key]}" for key in keys)
(root / "bundle.meta").write_text("\n".join(metadata) + "\n", encoding="utf-8")
PY

chmod 0444 "${stage}/bundle.meta" "${stage}/manifest.tsv"
validate_release "${stage}"
mv "${stage}" "${release_dir}"
trap - EXIT
printf '%s\n' \
  "codegen_key=${codegen_key}" \
  "compile_key=${compile_key}" \
  "target=${target_record}" \
  "bundle=${release_dir}" \
  "reused=false"
