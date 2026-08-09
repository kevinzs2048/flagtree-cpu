#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
libtriton_jit_dir="${repo_dir}/third_party/libtriton_jit"
output="${repo_dir}/artifacts/bench_cpp_wrapper"
overhead_output="${repo_dir}/artifacts/bench_wrapper_overhead"
fused_decode_output="${repo_dir}/artifacts/bench_fused_decode_aot"
fused_mlp_output="${repo_dir}/artifacts/bench_fused_mlp_aot"
attention_output="${repo_dir}/artifacts/bench_attention_aot"
f32_decode_output="${repo_dir}/artifacts/bench_f32_decode_aot"
f32_split_output="${repo_dir}/artifacts/bench_f32_split_aot"
argmax_output="${repo_dir}/artifacts/bench_argmax_aot"
norm_rope_output="${repo_dir}/artifacts/bench_norm_rope_aot"
roundeven_output="${repo_dir}/artifacts/bench_q8_roundeven_aot"
q4_roundeven_output="${repo_dir}/artifacts/bench_q4_roundeven_aot"
w8_wide_ab_output="${repo_dir}/artifacts/bench_w8_wide_codegen_ab"
w8_wide_schedule_ab_output="${repo_dir}/artifacts/bench_w8_wide_schedule_ab"
w8_vocab_parallel_output="${repo_dir}/artifacts/bench_w8_vocab_parallel"
q4_generated_pipeline_output="${repo_dir}/artifacts/bench_q4_generated_pipeline_aot"
q4_generated_decode_pipeline_output="${repo_dir}/artifacts/bench_q4_generated_decode_pipeline_aot"
q8_generated_pipeline_output="${repo_dir}/artifacts/bench_q8_generated_pipeline_aot"
q8_sve_fixed_ab_output="${repo_dir}/artifacts/bench_q8_sve_fixed_codegen_ab"
q4_pack_kleidiai_output="${repo_dir}/artifacts/bench_q4_pack_kleidiai_bf16"
w8_kleidiai_layout_output="${repo_dir}/artifacts/bench_w8_kleidiai_layout"
w4_kleidiai_layout_output="${repo_dir}/artifacts/bench_w4_kleidiai_layout"
qk_rmsnorm_native_output="${repo_dir}/artifacts/bench_qk_rmsnorm_native"
swiglu_quant_native_output="${repo_dir}/artifacts/bench_swiglu_quant_native"
w8_kai_bf16_lhs_pack_output="${repo_dir}/artifacts/bench_w8_kai_bf16_lhs_pack"
w8_kai_bf16_decode_output="${repo_dir}/artifacts/bench_w8_kai_bf16_decode_pipeline"
w8_kai_matrix_codegen_ab_output="${repo_dir}/artifacts/bench_w8_kai_matrix_codegen_ab"
w8_kai_bf16_prefill_output="${repo_dir}/artifacts/bench_w8_kai_bf16_prefill_pipeline"
w8_rhs_pack_reference_output="${repo_dir}/artifacts/libkai_w8_rhs_pack_reference.so"
kleidiai_root="${repo_dir}/third_party/llama.cpp-w4/build-kleidiai/_deps/kleidiai_download-src"
q4_pack_kleidiai_object="${repo_dir}/artifacts/kai_q4_pack_bench.o"
w8_lhs_pack_object="${repo_dir}/artifacts/kai_w8_lhs_pack_bench.o"
w8_rhs_pack_object="${repo_dir}/artifacts/kai_w8_rhs_pack_bench.o"
w8_rhs_pack_pic_object="${repo_dir}/artifacts/kai_w8_rhs_pack_reference_pic.o"
w8_matmul_object="${repo_dir}/artifacts/kai_w8_matmul_bench.o"
w8_bf16_lhs_pack_object="${repo_dir}/artifacts/kai_w8_bf16_lhs_pack_bench.o"
w8_i8mm_matmul_object="${repo_dir}/artifacts/kai_w8_i8mm_matmul_bench.o"
w4_lhs_pack_object="${repo_dir}/artifacts/kai_w4_lhs_pack_bench.o"
w4_rhs_pack_object="${repo_dir}/artifacts/kai_w4_rhs_pack_bench.o"
w4_matmul_object="${repo_dir}/artifacts/kai_w4_matmul_bench.o"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_cpp_wrapper.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${output}"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_wrapper_overhead.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${overhead_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_fused_decode_aot.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${fused_decode_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_fused_mlp_aot.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${fused_mlp_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_attention_aot.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${attention_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_f32_decode_aot.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${f32_decode_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_f32_split_aot.cpp" \
  -ldl -pthread \
  -o "${f32_split_output}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.2-a+dotprod+i8mm+bf16 \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_argmax_aot.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${argmax_output}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.6-a+bf16 \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_norm_rope_aot.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  -ldl -lffi -pthread \
  -o "${norm_rope_output}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.6-a+bf16 \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_qk_rmsnorm_native.cpp" \
  -ldl \
  -o "${qk_rmsnorm_native_output}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.6-a+bf16 \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_swiglu_quant_native.cpp" \
  -ldl \
  -o "${swiglu_quant_native_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_q8_roundeven_aot.cpp" \
  -ldl \
  -o "${roundeven_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_q4_roundeven_aot.cpp" \
  -ldl \
  -o "${q4_roundeven_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w8_wide_codegen_ab.cpp" \
  -ldl \
  -o "${w8_wide_ab_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w8_wide_schedule_ab.cpp" \
  -ldl \
  -o "${w8_wide_schedule_ab_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w8_vocab_parallel.cpp" \
  -ldl -pthread \
  -o "${w8_vocab_parallel_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_q4_generated_pipeline_aot.cpp" \
  -ldl \
  -o "${q4_generated_pipeline_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_q4_generated_decode_pipeline_aot.cpp" \
  -ldl \
  -o "${q4_generated_decode_pipeline_output}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.6-a+sve2+i8mm+bf16 \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_q8_generated_pipeline_aot.cpp" \
  -ldl \
  -o "${q8_generated_pipeline_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_q8_sve_fixed_codegen_ab.cpp" \
  -ldl \
  -o "${q8_sve_fixed_ab_output}"

gcc -std=c11 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.c" \
  -o "${q4_pack_kleidiai_object}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_q4_pack_kleidiai_bf16.cpp" \
  "${q4_pack_kleidiai_object}" \
  -ldl \
  -o "${q4_pack_kleidiai_output}"

gcc -std=c11 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c" \
  -o "${w8_lhs_pack_object}"

gcc -std=c11 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c" \
  -o "${w8_rhs_pack_object}"

gcc -std=c11 -O3 -DNDEBUG -fPIC -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c" \
  -o "${w8_rhs_pack_pic_object}"

g++ -std=c++20 -O3 -DNDEBUG -fPIC -shared -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  "${repo_dir}/benchmarks/cpp_wrapper/kai_w8_rhs_pack_reference.cpp" \
  "${w8_rhs_pack_pic_object}" \
  -o "${w8_rhs_pack_reference_output}"

gcc -std=c11 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c" \
  -o "${w8_matmul_object}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -I"${libtriton_jit_dir}/include" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w8_kleidiai_layout.cpp" \
  "${libtriton_jit_dir}/src/launch_hooks.cpp" \
  "${w8_lhs_pack_object}" "${w8_rhs_pack_object}" \
  "${w8_matmul_object}" \
  -ldl -lffi -pthread \
  -o "${w8_kleidiai_layout_output}"

gcc -std=c11 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c" \
  -o "${w4_lhs_pack_object}"

gcc -std=c11 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c" \
  -o "${w4_rhs_pack_object}"

gcc -std=c11 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c" \
  -o "${w4_matmul_object}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w4_kleidiai_layout.cpp" \
  "${w4_lhs_pack_object}" "${w4_rhs_pack_object}" \
  "${w4_matmul_object}" \
  -ldl \
  -o "${w4_kleidiai_layout_output}"

# GCC 12 ICEs in its O3 vectorizer on this upstream KAI source. O2 emits the
# same NEON pack loop latency on CIX and Apple gcc is Clang, so keep this one
# reference object at O2 rather than making the generated kernel depend on a
# host-specific compiler path.
gcc -std=c11 -O2 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.c" \
  -o "${w8_bf16_lhs_pack_object}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w8_kai_bf16_lhs_pack.cpp" \
  "${w8_bf16_lhs_pack_object}" \
  -ldl \
  -o "${w8_kai_bf16_lhs_pack_output}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.2-a+dotprod \
  -I"${kleidiai_root}" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w8_kai_bf16_decode_pipeline.cpp" \
  "${w8_bf16_lhs_pack_object}" "${w8_rhs_pack_object}" \
  "${w8_matmul_object}" \
  -ldl \
  -o "${w8_kai_bf16_decode_output}"

g++ -std=c++20 -O3 -DNDEBUG \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w8_kai_matrix_codegen_ab.cpp" \
  -ldl \
  -o "${w8_kai_matrix_codegen_ab_output}"

gcc -std=c11 -O3 -DNDEBUG -march=armv8.6-a+i8mm \
  -I"${kleidiai_root}" \
  -c "${kleidiai_root}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c" \
  -o "${w8_i8mm_matmul_object}"

g++ -std=c++20 -O3 -DNDEBUG -march=armv8.6-a+dotprod+i8mm \
  -I"${kleidiai_root}" \
  "${repo_dir}/benchmarks/cpp_wrapper/bench_w8_kai_bf16_prefill_pipeline.cpp" \
  "${w8_bf16_lhs_pack_object}" "${w8_rhs_pack_object}" \
  "${w8_i8mm_matmul_object}" \
  -ldl \
  -o "${w8_kai_bf16_prefill_output}"

printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n' \
  "${output}" "${overhead_output}" "${fused_decode_output}" \
  "${fused_mlp_output}" "${attention_output}" "${f32_decode_output}" \
  "${f32_split_output}" "${argmax_output}" "${norm_rope_output}" \
  "${roundeven_output}" "${q4_roundeven_output}" "${w8_wide_ab_output}" \
  "${w8_wide_schedule_ab_output}" "${w8_vocab_parallel_output}" \
  "${q4_generated_pipeline_output}" \
  "${q4_generated_decode_pipeline_output}" "${q8_generated_pipeline_output}" \
  "${q8_sve_fixed_ab_output}" "${q4_pack_kleidiai_output}" \
  "${w8_kleidiai_layout_output}" "${w4_kleidiai_layout_output}" \
  "${qk_rmsnorm_native_output}" \
  "${swiglu_quant_native_output}" \
  "${w8_kai_bf16_lhs_pack_output}" \
  "${w8_kai_bf16_decode_output}" "${w8_kai_bf16_prefill_output}" \
  "${w8_rhs_pack_reference_output}"
