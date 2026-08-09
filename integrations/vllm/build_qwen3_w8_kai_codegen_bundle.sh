#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
port_dir="${repo_dir}/ports/triton-cpu-3.7.2"
python_bin="${TRITON_TEST_PYTHON:-/home/kevin/venv-int8-clean/bin/python}"
venv_site="${TRITON_CPU_VENV_SITE:-/home/cix/venv-fep-e2e/lib/python3.11/site-packages}"
python_path="${port_dir}/python:${repo_dir}/third_party/FlagGems/src:${venv_site}"
bundle_root="${repo_dir}/artifacts/vllm-triton-backend/qwen3-w8-kai-codegen"
mkdir -p "${bundle_root}"

pin_prefix=()
if [[ -n "${TRITON_BUILD_CPU:-}" ]] && command -v taskset >/dev/null 2>&1; then
  pin_prefix=(taskset -c "${TRITON_BUILD_CPU}")
fi

target_record="$(
  env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
    TRITON_CPU_PYTHON="${port_dir}/python" TRITON_CPU_BACKEND=1 \
    "${python_bin}" -S - <<'PY'
import platform
import subprocess

from triton.compiler.compiler import make_backend
from triton.runtime.build import _find_compiler
from triton.runtime.driver import driver

target = driver.active.get_current_target()
backend = make_backend(target)
compiler = _find_compiler("c")
try:
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()[0]
except (OSError, IndexError):
    version = "unknown"
mode = "sve2_i8mm" if backend.use_sve2_i8mm() else "fixed_i8mm"
required = {"i8mm", "dotprod"}
if mode == "sve2_i8mm":
    required.update(("sve", "sve2"))
if "bf16" in backend.cpu_features:
    required.add("bf16")
fields = (
    platform.system().lower(),
    backend.cpu_arch,
    backend.cpu_name,
    mode,
    str(backend.sve_vector_bits),
    ",".join(sorted(required)),
    backend.hash(),
    compiler,
    version.replace("|", "/"),
)
print("|".join(fields))
PY
)"
IFS='|' read -r target_os target_arch target_cpu target_mode sve_vector_bits \
  required_features backend_hash target_compiler target_compiler_version \
  <<<"${target_record}"
target_tag="${target_os}-${target_arch}-${target_mode}-vl${sve_vector_bits}"
target_tag="$(printf '%s' "${target_tag}" | tr -c 'A-Za-z0-9._-' '_')"

hash_inputs() {
  CODEGEN_TARGET_RECORD="${target_record}" \
    CODEGEN_FIXED_I8MM="${TRITON_CPU_FIXED_I8MM:-0}" \
    CODEGEN_DISABLE_I8MM="${TRITON_CPU_DISABLE_SVE2_I8MM:-0}" \
    CODEGEN_COMPILE_KEY="${CODEGEN_COMPILE_KEY:-}" \
    "${python_bin}" -S - "$@" <<'PY'
import hashlib
import os
import pathlib
import sys

digest = hashlib.sha256()
for name in (
    "CODEGEN_TARGET_RECORD",
    "CODEGEN_FIXED_I8MM",
    "CODEGEN_DISABLE_I8MM",
    "CODEGEN_COMPILE_KEY",
):
    digest.update(name.encode())
    digest.update(b"=")
    digest.update(os.environ.get(name, "").encode())
    digest.update(b"\0")
for value in sys.argv[1:]:
    path = pathlib.Path(value)
    digest.update(str(path).encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
}

sha256_file() {
  "${python_bin}" -S - "$1" <<'PY'
import hashlib
import pathlib
import sys

print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

compile_key="$(CODEGEN_COMPILE_KEY= hash_inputs \
  "${repo_dir}/benchmarks/bench_w8_kai_bf16_lhs_pack_codegen.py" \
  "${repo_dir}/benchmarks/bench_w8_kleidiai_layout_codegen.py" \
  "${repo_dir}/benchmarks/bench_w8_prefill_kai_layout_codegen.py" \
  "${port_dir}/third_party/cpu/backend/compiler.py" \
  "${port_dir}/third_party/cpu/backend/target_info.py" \
  "${port_dir}/python/triton/runtime/build.py" \
  "${port_dir}/third_party/cpu/lib/TritonCPUTransforms/ConvertDotProduct.cpp" \
  "${port_dir}/third_party/cpu/lib/TritonCPUTransforms/ConvertDotOp/ConvertDotToSVE2I8MM.cpp" \
  "${port_dir}/python/triton/_C/libtriton.so")"
codegen_key="$(CODEGEN_COMPILE_KEY="${compile_key}" hash_inputs \
  "${repo_dir}/integrations/vllm/build_qwen3_w8_kai_codegen_bundle.sh")"
cache_root="${repo_dir}/artifacts/aot-vllm-kai-w8-3.7-${target_tag}-${compile_key}"
release_parent="${bundle_root}/${target_tag}"
release_dir="${release_parent}/${codegen_key}"
mkdir -p "${release_parent}"

validate_release() {
  local candidate="$1"
  "${python_bin}" -S - "${candidate}" "${codegen_key}" \
      "${compile_key}" "${target_os}" "${target_arch}" \
      "${target_mode}" "${sve_vector_bits}" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    "format_version": "2",
    "backend_abi": "2",
    "codegen_key": sys.argv[2],
    "compile_key": sys.argv[3],
    "target_os": sys.argv[4],
    "target_arch": sys.argv[5],
    "target_mode": sys.argv[6],
    "sve_vector_bits": sys.argv[7],
    "kai_rhs_abi": "qsi8cxp4x8-nr4-kr8-sr1",
    "kai_lhs_abi": "qai8dxp-mr1-mr4-kr8-sr1",
}

def properties(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"invalid metadata line in {path}")
        result[key] = value
    return result

metadata = properties(root / "bundle.meta")
for key, value in expected.items():
    if metadata.get(key) != value:
        raise ValueError(
            f"bundle metadata mismatch for {key}: "
            f"{metadata.get(key)!r} != {value!r}"
        )
manifest = (root / "manifest.tsv").read_text(encoding="utf-8").splitlines()
header = [
    "#k", "n", "shape", "pack_sha256", "matrix_sha256",
    "pack_m16_sha256", "prefill_m4_sha256", "prefill_m8_sha256",
    "prefill_m12_sha256", "prefill_m16_sha256",
]
if not manifest or manifest[0].split("\t") != header:
    raise ValueError("invalid bundle manifest header")
files = (
    ("pack_sha256", "_pack_lhs_qai8dxp_bf16_kernel.so"),
    ("matrix_sha256", "_kai_w8_layout_pointer_kernel.so"),
    ("pack_m16_sha256", "_pack_lhs_qai8dxp_bf16_mr4_kernel.so"),
    ("prefill_m4_sha256", "_kai_w8_prefill_m4_kernel.so"),
    ("prefill_m8_sha256", "_kai_w8_prefill_m8_kernel.so"),
    ("prefill_m12_sha256", "_kai_w8_prefill_m12_kernel.so"),
    ("prefill_m16_sha256", "_kai_w8_prefill_kernel.so"),
)
shapes = 0
seen = set()
for line in manifest[1:]:
    values = line.split("\t")
    if len(values) != len(header):
        raise ValueError("invalid bundle manifest row")
    row = dict(zip(header, values))
    canonical = f"k{int(row['#k'])}-n{int(row['n'])}"
    if row["shape"] != canonical:
        raise ValueError("non-canonical bundle shape")
    if canonical in seen:
        raise ValueError("duplicate bundle shape")
    seen.add(canonical)
    for column, filename in files:
        path = root / canonical / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular bundle object: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row[column]:
            raise ValueError(f"bundle object digest mismatch: {path}")
    shapes += 1
if shapes != int(metadata["shapes"]):
    raise ValueError("bundle shape count mismatch")
PY
}

if [[ -f "${release_dir}/bundle.meta" ]]; then
  if ! validate_release "${release_dir}"; then
    printf 'refusing corrupt generated W8 release: %s\n' \
      "${release_dir}" >&2
    exit 1
  fi
  printf '%s\n' \
    "codegen_key=${codegen_key}" \
    "compile_key=${compile_key}" \
    "target=${target_record}" \
    "bundle=${release_dir}" \
    "reused=true"
  exit 0
fi
bundle_dir="$(mktemp -d "${release_parent}/.staging-${codegen_key}.XXXXXX")"
cleanup_bundle() {
  rm -rf -- "${bundle_dir}"
}
trap cleanup_bundle EXIT

if [[ -n "${OBJDUMP:-}" ]]; then
  objdump_bin="${OBJDUMP}"
elif command -v llvm-objdump >/dev/null 2>&1; then
  objdump_bin="$(command -v llvm-objdump)"
elif command -v gobjdump >/dev/null 2>&1; then
  objdump_bin="$(command -v gobjdump)"
else
  objdump_bin="$(command -v objdump || true)"
fi
if [[ -z "${objdump_bin}" ]]; then
  printf 'no objdump implementation found; set OBJDUMP explicitly\n' >&2
  exit 1
fi

disassemble_symbol() {
  local object_path="$1"
  local symbol="$2"
  local output
  if output="$("${objdump_bin}" -d --disassemble="${symbol}" \
      "${object_path}" 2>/dev/null)" && [[ -n "${output}" ]]; then
    printf '%s\n' "${output}"
    return
  fi
  if output="$("${objdump_bin}" -d --disassemble-symbols="${symbol}" \
      "${object_path}" 2>/dev/null)" && [[ -n "${output}" ]]; then
    printf '%s\n' "${output}"
    return
  fi
  "${objdump_bin}" -d "${object_path}"
}

compile_pack() {
  local k="$1"
  local cache_dir="${cache_root}/pack-k${k}"
  if ! find "${cache_dir}" -type f \
      -name '_pack_lhs_qai8dxp_bf16_kernel.so' -print -quit \
      2>/dev/null | grep -q .; then
    env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
      TRITON_CPU_PYTHON="${port_dir}/python" \
      TRITON_CPU_BACKEND=1 TRITON_CACHE_DIR="${cache_dir}" \
      OMP_NUM_THREADS=1 "${pin_prefix[@]}" "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_w8_kai_bf16_lhs_pack_codegen.py" \
      --m 1 --k "${k}" --mr 1 --compile-only >&2 || return
  fi
  find "${cache_dir}" -type f \
    -name '_pack_lhs_qai8dxp_bf16_kernel.so' -print -quit
}

compile_matrix() {
  local k="$1"
  local n="$2"
  local cache_dir="${cache_root}/matrix-k${k}-n${n}"
  if ! find "${cache_dir}" -type f \
      -name '_kai_w8_layout_pointer_kernel.so' -print -quit \
      2>/dev/null | grep -q .; then
    env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
      TRITON_CPU_PYTHON="${port_dir}/python" \
      TRITON_CPU_BACKEND=1 TRITON_CACHE_DIR="${cache_dir}" \
      OMP_NUM_THREADS=1 "${pin_prefix[@]}" "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_w8_kleidiai_layout_codegen.py" \
      --k "${k}" --n "${n}" --unroll 2 --mode pointer \
      --output-bf16 --compile-only >&2 || return
  fi
  find "${cache_dir}" -type f \
    -name '_kai_w8_layout_pointer_kernel.so' -print -quit
}

compile_pack_m16() {
  local k="$1"
  local cache_dir="${cache_root}/pack-m16-k${k}"
  if ! find "${cache_dir}" -type f \
      -name '_pack_lhs_qai8dxp_bf16_mr4_kernel.so' -print -quit \
      2>/dev/null | grep -q .; then
    env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
      TRITON_CPU_PYTHON="${port_dir}/python" \
      TRITON_CPU_BACKEND=1 TRITON_CACHE_DIR="${cache_dir}" \
      OMP_NUM_THREADS=1 "${pin_prefix[@]}" "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_w8_kai_bf16_lhs_pack_codegen.py" \
      --m 16 --k "${k}" --mr 4 --schedule panel4 \
      --vector-reduction --narrow-quant --compile-only >&2 || return
  fi
  find "${cache_dir}" -type f \
    -name '_pack_lhs_qai8dxp_bf16_mr4_kernel.so' -print -quit
}

compile_prefill() {
  local m="$1"
  local k="$2"
  local n="$3"
  local symbol
  if [[ "${m}" == 4 || "${m}" == 8 ]]; then
    symbol="_kai_w8_prefill_short_tail_kernel.so"
  elif [[ "${m}" == 12 ]]; then
    symbol="_kai_w8_prefill_m12_tail_kernel.so"
  elif [[ "${m}" == 16 ]]; then
    symbol="_kai_w8_prefill_kernel.so"
  else
    printf 'invalid generated W8 prefill M: %s\n' "${m}" >&2
    return 1
  fi
  local cache_dir="${cache_root}/prefill-m${m}-k${k}-n${n}"
  if ! find "${cache_dir}" -type f \
      -name "${symbol}" -print -quit \
      2>/dev/null | grep -q .; then
    env PYTHONPATH="${python_path}" TRITON_BACKENDS_IN_TREE=1 \
      TRITON_CPU_PYTHON="${port_dir}/python" \
      TRITON_CPU_BACKEND=1 TRITON_CACHE_DIR="${cache_dir}" \
      OMP_NUM_THREADS=1 "${pin_prefix[@]}" "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_w8_prefill_kai_layout_codegen.py" \
      --m "${m}" --n "${n}" --k "${k}" --output-bf16 \
      --warmup 0 --iters 1 >&2 || return
  fi
  find "${cache_dir}" -type f \
    -name "${symbol}" -print -quit
}

audit_pack() {
  local object_path="$1"
  local body round_count stack_count call_count
  body="$(disassemble_symbol "${object_path}" \
    _pack_lhs_qai8dxp_bf16_kernel)"
  round_count="$(awk '$3 == "fcvtns" { count++ } END { print count + 0 }' \
    <<<"${body}")"
  stack_count="$(awk '/\[sp/ { count++ } END { print count + 0 }' \
    <<<"${body}")"
  call_count="$(awk '$3 == "bl" || $3 == "blr" { count++ } \
    END { print count + 0 }' <<<"${body}")"
  if [[ "${round_count}" != 3 || "${stack_count}" != 0 \
      || "${call_count}" != 0 ]]; then
    printf 'KAI W8 pack audit failed: fcvtns=%s stack=%s calls=%s %s\n' \
      "${round_count}" "${stack_count}" "${call_count}" \
      "${object_path}" >&2
    return 1
  fi
}

audit_matrix() {
  local object_path="$1"
  local body sdot_count addp_count ld1r_count lane_count stack_count call_count
  body="$(disassemble_symbol "${object_path}" \
    _kai_w8_layout_pointer_kernel)"
  sdot_count="$(awk '$3 == "sdot" { count++ } END { print count + 0 }' \
    <<<"${body}")"
  addp_count="$(awk '$3 == "addp" { count++ } END { print count + 0 }' \
    <<<"${body}")"
  ld1r_count="$(awk '$3 == "ld1r" { count++ } END { print count + 0 }' \
    <<<"${body}")"
  lane_count="$(awk '$3 == "mov" && $0 ~ /\.d\[/ { count++ } \
    END { print count + 0 }' <<<"${body}")"
  stack_count="$(awk '/\[sp/ { count++ } END { print count + 0 }' \
    <<<"${body}")"
  call_count="$(awk '$3 == "bl" || $3 == "blr" { count++ } \
    END { print count + 0 }' <<<"${body}")"
  if [[ "${sdot_count}" != 16 || "${addp_count}" != 1 \
      || "${ld1r_count}" != 10 || "${lane_count}" != 0 \
      || "${stack_count}" != 0 || "${call_count}" != 0 ]]; then
    printf 'KAI W8 matrix audit failed: sdot=%s addp=%s ld1r=%s lane=%s stack=%s calls=%s %s\n' \
      "${sdot_count}" "${addp_count}" "${ld1r_count}" "${lane_count}" \
      "${stack_count}" "${call_count}" "${object_path}" >&2
    return 1
  fi
}

audit_pack_m16() {
  local object_path="$1"
  local body round_count stack_count call_count
  body="$(disassemble_symbol "${object_path}" \
    _pack_lhs_qai8dxp_bf16_mr4_kernel)"
  round_count="$(awk '$3 == "fcvtns" { count++ } END { print count + 0 }' \
    <<<"${body}")"
  stack_count="$(awk '/\[sp/ { count++ } END { print count + 0 }' \
    <<<"${body}")"
  call_count="$(awk '$3 == "bl" || $3 == "blr" { count++ } \
    END { print count + 0 }' <<<"${body}")"
  if [[ "${round_count}" != 9 || "${stack_count}" != 0 \
      || "${call_count}" != 0 ]]; then
    printf 'KAI W8 M16 pack audit failed: fcvtns=%s stack=%s calls=%s %s\n' \
      "${round_count}" "${stack_count}" "${call_count}" \
      "${object_path}" >&2
    return 1
  fi
}

audit_prefill() {
  local m="$1"
  local object_path="$2"
  local symbol expected_smmla max_stack
  if [[ "${m}" == 4 || "${m}" == 8 ]]; then
    symbol="_kai_w8_prefill_short_tail_kernel"
    max_stack=0
  elif [[ "${m}" == 12 ]]; then
    symbol="_kai_w8_prefill_m12_tail_kernel"
    max_stack=0
  elif [[ "${m}" == 16 ]]; then
    symbol="_kai_w8_prefill_kernel"
    max_stack=8
  else
    printf 'invalid generated W8 prefill audit M: %s\n' "${m}" >&2
    return 1
  fi
  expected_smmla=$((4 * m))
  local body smmla_count hot_stack_count stack_count call_count
  body="$(disassemble_symbol "${object_path}" "${symbol}")"
  smmla_count="$(awk '$3 == "smmla" { count++ } END { print count + 0 }' \
    <<<"${body}")"
  hot_stack_count="$(awk '
    $3 == "smmla" { if (!first) first = NR; last = NR }
    { lines[NR] = $0 }
    END {
      for (line = first; line <= last; ++line)
        if (lines[line] ~ /\[sp/) count++
      print count + 0
    }' <<<"${body}")"
  stack_count="$(awk '/\[sp/ { count++ } END { print count + 0 }' \
    <<<"${body}")"
  call_count="$(awk '$3 == "bl" || $3 == "blr" { count++ } \
    END { print count + 0 }' <<<"${body}")"
  if [[ "${smmla_count}" != "${expected_smmla}" \
      || "${hot_stack_count}" != 0 || "${stack_count}" -gt "${max_stack}" \
      || "${call_count}" != 0 ]]; then
    printf 'KAI W8 M%s prefill audit failed: smmla=%s hot_stack=%s total_stack=%s calls=%s %s\n' \
      "${m}" "${smmla_count}" "${hot_stack_count}" "${stack_count}" \
      "${call_count}" "${object_path}" >&2
    return 1
  fi
}

shapes=(
  # Existing per-projection and audit shapes.
  "1024 1024"
  "1024 2048"
  "1024 3072"
  "1024 4096"
  # Qwen3-0.6B production fused/row projections:
  # qkv is 1024x4096 above; these are o, gate_up, and down.
  "2048 1024"
  "1024 6144"
  "3072 1024"
  # Qwen3-1.7B production qkv, o, gate_up, and down projections.
  "2048 4096"
  "2048 2048"
  "2048 12288"
  "6144 2048"
  # Qwen3-4B per-projection and production row projections.
  "2560 1024"
  "2560 4096"
  "2560 9728"
  "4096 2560"
  "9728 2560"
  # Qwen3-4B production fused qkv and gate_up projections.
  "2560 6144"
  "2560 19456"
)

printf '%s\n' \
  $'#k\tn\tshape\tpack_sha256\tmatrix_sha256\tpack_m16_sha256\tprefill_m4_sha256\tprefill_m8_sha256\tprefill_m12_sha256\tprefill_m16_sha256' \
  >"${bundle_dir}/manifest.tsv"
for shape in "${shapes[@]}"; do
  read -r k n <<<"${shape}"
  pack_so="$(compile_pack "${k}")"
  matrix_so="$(compile_matrix "${k}" "${n}")"
  pack_m16_so="$(compile_pack_m16 "${k}")"
  prefill_m4_so="$(compile_prefill 4 "${k}" "${n}")"
  prefill_m8_so="$(compile_prefill 8 "${k}" "${n}")"
  prefill_m12_so="$(compile_prefill 12 "${k}" "${n}")"
  prefill_m16_so="$(compile_prefill 16 "${k}" "${n}")"
  audit_pack "${pack_so}"
  audit_matrix "${matrix_so}"
  audit_pack_m16 "${pack_m16_so}"
  audit_prefill 4 "${prefill_m4_so}"
  audit_prefill 8 "${prefill_m8_so}"
  audit_prefill 12 "${prefill_m12_so}"
  audit_prefill 16 "${prefill_m16_so}"
  shape_name="k${k}-n${n}"
  shape_dir="${bundle_dir}/${shape_name}"
  mkdir -p "${shape_dir}"
  cp "${pack_so}" "${shape_dir}/_pack_lhs_qai8dxp_bf16_kernel.so"
  cp "${matrix_so}" "${shape_dir}/_kai_w8_layout_pointer_kernel.so"
  cp "${pack_m16_so}" \
    "${shape_dir}/_pack_lhs_qai8dxp_bf16_mr4_kernel.so"
  cp "${prefill_m4_so}" "${shape_dir}/_kai_w8_prefill_m4_kernel.so"
  cp "${prefill_m8_so}" "${shape_dir}/_kai_w8_prefill_m8_kernel.so"
  cp "${prefill_m12_so}" "${shape_dir}/_kai_w8_prefill_m12_kernel.so"
  cp "${prefill_m16_so}" "${shape_dir}/_kai_w8_prefill_kernel.so"
  chmod 0755 "${shape_dir}"/*.so
  pack_hash="$(sha256_file \
    "${shape_dir}/_pack_lhs_qai8dxp_bf16_kernel.so")"
  matrix_hash="$(sha256_file \
    "${shape_dir}/_kai_w8_layout_pointer_kernel.so")"
  pack_m16_hash="$(sha256_file \
    "${shape_dir}/_pack_lhs_qai8dxp_bf16_mr4_kernel.so")"
  prefill_m4_hash="$(sha256_file \
    "${shape_dir}/_kai_w8_prefill_m4_kernel.so")"
  prefill_m8_hash="$(sha256_file \
    "${shape_dir}/_kai_w8_prefill_m8_kernel.so")"
  prefill_m12_hash="$(sha256_file \
    "${shape_dir}/_kai_w8_prefill_m12_kernel.so")"
  prefill_m16_hash="$(sha256_file \
    "${shape_dir}/_kai_w8_prefill_kernel.so")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${k}" "${n}" "${shape_name}" "${pack_hash}" "${matrix_hash}" \
    "${pack_m16_hash}" "${prefill_m4_hash}" "${prefill_m8_hash}" \
    "${prefill_m12_hash}" "${prefill_m16_hash}" \
    >>"${bundle_dir}/manifest.tsv"
done

{
  printf '%s\n' \
    "format_version=2" \
    "backend_abi=2" \
    "codegen_key=${codegen_key}" \
    "compile_key=${compile_key}" \
    "target_os=${target_os}" \
    "target_arch=${target_arch}" \
    "target_cpu=${target_cpu}" \
    "target_mode=${target_mode}" \
    "sve_vector_bits=${sve_vector_bits}" \
    "required_features=${required_features}" \
    "backend_hash=${backend_hash}" \
    "compiler=${target_compiler}" \
    "compiler_version=${target_compiler_version}" \
    "kai_rhs_abi=qsi8cxp4x8-nr4-kr8-sr1" \
    "kai_lhs_abi=qai8dxp-mr1-mr4-kr8-sr1" \
    "shapes=${#shapes[@]}" \
    "matrix_contract=16_sdot_1_addp_10_ld1r_0_lane_0_stack_0_call" \
    "pack_contract=3_fcvtns_0_stack_0_call" \
    "prefill_contract=m4_16_m8_32_m12_48_m16_64_smmla_0_hot_stack" \
    "pack_m16_contract=9_fcvtns_0_stack_0_call"
} >"${bundle_dir}/bundle.meta"

validate_release "${bundle_dir}"
find "${bundle_dir}" -type f -exec chmod 0444 {} +

publish_status=0
"${python_bin}" -S - "${bundle_dir}" "${release_dir}" <<'PY' \
  || publish_status=$?
import os
import pathlib
import sys

staging = pathlib.Path(sys.argv[1])
release = pathlib.Path(sys.argv[2])
try:
    os.rename(staging, release)
except OSError:
    if (release / "bundle.meta").is_file():
        raise SystemExit(17)
    raise
PY
if [[ "${publish_status}" == 17 ]]; then
  # A concurrent builder published the same content-addressed release.
  if ! validate_release "${release_dir}"; then
    printf 'concurrent generated W8 release failed validation: %s\n' \
      "${release_dir}" >&2
    exit 1
  fi
  rm -rf -- "${bundle_dir}"
elif [[ "${publish_status}" != 0 ]]; then
  exit "${publish_status}"
fi
bundle_dir=""
trap - EXIT

printf '%s\n' \
  "codegen_key=${codegen_key}" \
  "compile_key=${compile_key}" \
  "target=${target_record}" \
  "bundle=${release_dir}" \
  "shapes=${#shapes[@]}" \
  "matrix_contract=16_sdot_1_addp_10_ld1r_0_lane_0_stack_0_call" \
  "pack_contract=3_fcvtns_0_stack_0_call" \
  "prefill_contract=m4_16_m8_32_m12_48_m16_64_smmla_0_hot_stack" \
  "pack_m16_contract=9_fcvtns_0_stack_0_call"
