#include <arm_neon.h>
#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {

using ThreePointerKernel = void (*)(void*, void*, void*, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t, uint32_t);
using FourPointerKernel = void (*)(void*, void*, void*, void*, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t);
using TwoPointerKernel = void (*)(void*, void*, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t);

struct SharedObject {
  explicit SharedObject(const std::string& path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      throw std::runtime_error(dlerror());
    }
  }

  ~SharedObject() {
    if (handle) {
      dlclose(handle);
    }
  }

  template <typename Function>
  Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle, name);
    if (const char* error = dlerror()) {
      throw std::runtime_error(error);
    }
    return reinterpret_cast<Function>(address);
  }

  void* handle = nullptr;
};

std::string dirname(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

struct TripleTiming {
  double first;
  double second;
  double third;
};

template <typename Function>
double measure_batch_us(Function&& function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    function();
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}

// Rotate the order of the three implementations in every batch.  This avoids
// attributing frequency ramp or thermal drift to whichever path runs last.
template <typename First, typename Second, typename Third>
TripleTiming median_three_us(First&& first, Second&& second, Third&& third,
                             int iterations, int batches = 15) {
  for (int i = 0; i < iterations; ++i) {
    first();
    second();
    third();
  }
  std::vector<double> first_samples;
  std::vector<double> second_samples;
  std::vector<double> third_samples;
  first_samples.reserve(batches);
  second_samples.reserve(batches);
  third_samples.reserve(batches);
  for (int batch = 0; batch < batches; ++batch) {
    auto measure_first = [&] {
      first_samples.push_back(measure_batch_us(first, iterations));
    };
    auto measure_second = [&] {
      second_samples.push_back(measure_batch_us(second, iterations));
    };
    auto measure_third = [&] {
      third_samples.push_back(measure_batch_us(third, iterations));
    };
    if (batch % 3 == 0) {
      measure_first();
      measure_second();
      measure_third();
    } else if (batch % 3 == 1) {
      measure_second();
      measure_third();
      measure_first();
    } else {
      measure_third();
      measure_first();
      measure_second();
    }
  }
  std::sort(first_samples.begin(), first_samples.end());
  std::sort(second_samples.begin(), second_samples.end());
  std::sort(third_samples.begin(), third_samples.end());
  return {first_samples[first_samples.size() / 2],
          second_samples[second_samples.size() / 2],
          third_samples[third_samples.size() / 2]};
}

struct F32x8 {
  float32x4_t lo;
  float32x4_t hi;
};

F32x8 load_bf16(const uint16_t* pointer) {
  const auto packed =
      vld1q_bf16(reinterpret_cast<const bfloat16_t*>(pointer));
  return {vcvt_f32_bf16(vget_low_bf16(packed)),
          vcvt_f32_bf16(vget_high_bf16(packed))};
}

float32x4_t load_bf16x4(const uint16_t* pointer) {
  return vcvt_f32_bf16(
      vld1_bf16(reinterpret_cast<const bfloat16_t*>(pointer)));
}

void store_bf16(uint16_t* pointer, F32x8 value) {
  const auto packed = vcombine_bf16(vcvt_bf16_f32(value.lo),
                                    vcvt_bf16_f32(value.hi));
  vst1q_bf16(reinterpret_cast<bfloat16_t*>(pointer), packed);
}

void store_bf16x4(uint16_t* pointer, float32x4_t value) {
  vst1_bf16(reinterpret_cast<bfloat16_t*>(pointer), vcvt_bf16_f32(value));
}

float tile_square_sum(F32x8 a, F32x8 b) {
  const float32x4_t a2 = vmulq_f32(a.lo, a.lo);
  const float32x4_t a3 = vmulq_f32(a.hi, a.hi);
  const float32x4_t b2 = vmulq_f32(b.lo, b.lo);
  const float32x4_t b3 = vmulq_f32(b.hi, b.hi);
  return vaddvq_f32(a2) + vaddvq_f32(a3) + vaddvq_f32(b2) +
         vaddvq_f32(b3);
}

__attribute__((noinline)) void neon_rms(const uint16_t* input,
                                        const uint16_t* weight,
                                        uint16_t* output, int32_t size,
                                        float eps) {
  float square_sum = 0.0f;
  for (int32_t offset = 0; offset < size; offset += 16) {
    square_sum += tile_square_sum(load_bf16(input + offset),
                                  load_bf16(input + offset + 8));
  }
  const float rrms = 1.0f / std::sqrt(square_sum / size + eps);
  const float32x4_t scale = vdupq_n_f32(rrms);
  for (int32_t offset = 0; offset < size; offset += 8) {
    F32x8 x = load_bf16(input + offset);
    const F32x8 w = load_bf16(weight + offset);
    x.lo = vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(x.lo, scale)));
    x.hi = vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(x.hi, scale)));
    store_bf16(output + offset,
               {vmulq_f32(x.lo, w.lo), vmulq_f32(x.hi, w.hi)});
  }
}

// Match vLLM's current residual semantics: normalize the unrounded FP32 sum,
// but materialize that sum separately as BF16 in the residual output.
__attribute__((noinline)) void neon_vllm_fused_rms(
    uint16_t* input, uint16_t* residual, const uint16_t* weight, int32_t size,
    float eps) {
  float square_sum = 0.0f;
  for (int32_t offset = 0; offset < size; offset += 16) {
    F32x8 x0 = load_bf16(input + offset);
    F32x8 x1 = load_bf16(input + offset + 8);
    const F32x8 r0 = load_bf16(residual + offset);
    const F32x8 r1 = load_bf16(residual + offset + 8);
    x0 = {vaddq_f32(x0.lo, r0.lo), vaddq_f32(x0.hi, r0.hi)};
    x1 = {vaddq_f32(x1.lo, r1.lo), vaddq_f32(x1.hi, r1.hi)};
    square_sum += tile_square_sum(x0, x1);
  }

  const float rrms = 1.0f / std::sqrt(square_sum / size + eps);
  const float32x4_t scale = vdupq_n_f32(rrms);
  for (int32_t offset = 0; offset < size; offset += 8) {
    F32x8 x = load_bf16(input + offset);
    const F32x8 r = load_bf16(residual + offset);
    const F32x8 w = load_bf16(weight + offset);
    x = {vaddq_f32(x.lo, r.lo), vaddq_f32(x.hi, r.hi)};
    store_bf16(residual + offset, x);
    F32x8 normalized = {
        vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(x.lo, scale))),
        vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(x.hi, scale)))};
    store_bf16(input + offset,
               {vmulq_f32(normalized.lo, w.lo),
                vmulq_f32(normalized.hi, w.hi)});
  }
}

__attribute__((noinline)) void neon_rope(
    uint16_t* q, uint16_t* k, const int64_t* positions,
    const uint16_t* cos_sin_cache, int q_heads, int kv_heads, int head_dim) {
  const int half = head_dim / 2;
  const uint16_t* cache = cos_sin_cache + positions[0] * head_dim;
  const uint16_t* cosine = cache;
  const uint16_t* sine = cache + half;
  for (int head = 0; head < q_heads + kv_heads; ++head) {
    uint16_t* row = head < q_heads ? q + head * head_dim
                                   : k + (head - q_heads) * head_dim;
    for (int offset = 0; offset < half; offset += 4) {
      const float32x4_t first = load_bf16x4(row + offset);
      const float32x4_t second = load_bf16x4(row + half + offset);
      const float32x4_t c = load_bf16x4(cosine + offset);
      const float32x4_t s = load_bf16x4(sine + offset);
      store_bf16x4(row + offset,
                   vsubq_f32(vmulq_f32(first, c), vmulq_f32(second, s)));
      store_bf16x4(row + half + offset,
                   vaddq_f32(vmulq_f32(first, s), vmulq_f32(second, c)));
    }
  }
}

float32x4_t sleef_u10_exp_inline(float32x4_t value) {
  const int32x4_t exponent =
      vcvtnq_s32_f32(vmulq_n_f32(value, 1.4426950408889634f));
  const float32x4_t exponent_f = vcvtq_f32_s32(exponent);
  float32x4_t reduced =
      vfmaq_n_f32(value, exponent_f, -0.693145751953125f);
  reduced = vfmaq_n_f32(reduced, exponent_f, -1.428606765330187e-6f);

  float32x4_t polynomial = vdupq_n_f32(0.00019852761761285365f);
  polynomial = vfmaq_f32(vdupq_n_f32(0.0013930435525253415f), polynomial,
                         reduced);
  polynomial = vfmaq_f32(vdupq_n_f32(0.008333360776305199f), polynomial,
                         reduced);
  polynomial = vfmaq_f32(vdupq_n_f32(0.041666485369205475f), polynomial,
                         reduced);
  polynomial = vfmaq_f32(vdupq_n_f32(0.1666666716337204f), polynomial,
                         reduced);
  polynomial =
      vfmaq_f32(vdupq_n_f32(0.5f), polynomial, reduced);
  float32x4_t result = vfmaq_f32(reduced, vmulq_f32(reduced, reduced),
                                 polynomial);
  result = vaddq_f32(vdupq_n_f32(1.0f), result);

  const int32x4_t half_exponent = vshrq_n_s32(exponent, 1);
  const int32x4_t pow0_bits =
      vshlq_n_s32(vaddq_s32(half_exponent, vdupq_n_s32(127)), 23);
  const int32x4_t pow1_bits = vshlq_n_s32(
      vaddq_s32(vsubq_s32(exponent, half_exponent), vdupq_n_s32(127)), 23);
  result = vmulq_f32(result, vreinterpretq_f32_s32(pow0_bits));
  result = vmulq_f32(result, vreinterpretq_f32_s32(pow1_bits));
  result = vbslq_f32(vcltq_f32(value, vdupq_n_f32(-104.0f)),
                     vdupq_n_f32(0.0f), result);
  return vbslq_f32(vcgtq_f32(value, vdupq_n_f32(100.0f)),
                   vdupq_n_f32(std::numeric_limits<float>::infinity()),
                   result);
}

__attribute__((noinline)) void neon_swiglu(const uint16_t* gate_up,
                                           uint16_t* output, int32_t size) {
  for (int32_t offset = 0; offset < size; offset += 4) {
    const float32x4_t gate = load_bf16x4(gate_up + offset);
    const float32x4_t up = load_bf16x4(gate_up + size + offset);
    const float32x4_t denominator = vaddq_f32(
        vdupq_n_f32(1.0f), sleef_u10_exp_inline(vnegq_f32(gate)));
    float32x4_t silu = vdivq_f32(gate, denominator);
    silu = vcvt_f32_bf16(vcvt_bf16_f32(silu));
    store_bf16x4(output + offset, vmulq_f32(silu, up));
  }
}

size_t mismatches(const std::vector<uint16_t>& a,
                  const std::vector<uint16_t>& b) {
  size_t count = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    count += a[i] != b[i];
  }
  return count;
}

void print_result(const char* name, double direct_us, double wrapper_us,
                  double c_us, size_t mismatch, size_t elements) {
  std::cout << name << "_direct_aot_us=" << direct_us << '\n'
            << name << "_libtriton_jit_us=" << wrapper_us << '\n'
            << name << "_acle_c_us=" << c_us << '\n'
            << name << "_direct_over_c=" << direct_us / c_us << "x\n"
            << name << "_jit_over_direct=" << wrapper_us / direct_us
            << "x\n"
            << name << "_jit_minus_direct_us=" << wrapper_us - direct_us
            << '\n'
            << name << "_mismatch=" << mismatch << '/' << elements << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " RMS_SO FUSED_RMS_SO ROPE_SO SWIGLU_SO\n";
    return 2;
  }

  try {
    constexpr int size = 1024;
    constexpr float eps = 1.0e-6f;
    std::vector<uint16_t> x(size), weight(size), triton_out(size), c_out(size),
        wrapper_out(size);
    for (int i = 0; i < size; ++i) {
      x[i] = to_bf16(std::sin(i * 0.017f));
      weight[i] = to_bf16(std::cos(i * 0.013f));
    }

    SharedObject rms_lib(argv[1]);
    ThreePointerKernel rms =
        rms_lib.symbol<ThreePointerKernel>("_rms_norm_aot_kernel");
    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> rms_wrapper(
        dirname(argv[1]), "_rms_norm_aot_kernel");
    void* rms_x_pointer = x.data();
    void* rms_weight_pointer = weight.data();
    void* rms_wrapper_out_pointer = wrapper_out.data();
    uint16_t* rms_c_out_pointer = c_out.data();
    void* rms_args[] = {&rms_x_pointer, &rms_weight_pointer,
                        &rms_wrapper_out_pointer};
    auto run_rms = [&] {
      asm volatile("" : : "r"(x.data()) : "memory");
      rms(x.data(), weight.data(), triton_out.data(), 0, 0, 0, 1, 1, 1);
    };
    auto run_rms_wrapper = [&] {
      rms_wrapper.launch_with_signature(1, 1, 1, 1, nullptr, rms_args,
                                        "*bf16,*bf16,*bf16", 3);
    };
    auto run_rms_c = [&] {
      asm volatile("" : : "r"(x.data()) : "memory");
      neon_rms(x.data(), weight.data(), rms_c_out_pointer, size, eps);
    };
    run_rms();
    run_rms_c();
    run_rms_wrapper();
    const size_t rms_mismatch = mismatches(triton_out, c_out) +
                                mismatches(triton_out, wrapper_out);
    // Time all implementations on the same addresses.  BF16 vector allocation
    // alignment otherwise moves a tiny kernel by more than the wrapper cost.
    rms_wrapper_out_pointer = triton_out.data();
    rms_c_out_pointer = triton_out.data();
    const TripleTiming rms_timing =
        median_three_us(run_rms, run_rms_wrapper, run_rms_c, 20000);

    SharedObject fused_lib(argv[2]);
    ThreePointerKernel fused = fused_lib.symbol<ThreePointerKernel>(
        "_vllm_fused_add_rms_aot_kernel");
    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> fused_wrapper(
        dirname(argv[2]), "_vllm_fused_add_rms_aot_kernel");
    std::vector<uint16_t> fused_x(size, to_bf16(0.0f));
    std::vector<uint16_t> fused_residual = x;
    std::vector<uint16_t> zero_weight(size, to_bf16(0.0f));
    std::vector<uint16_t> c_x = fused_x;
    std::vector<uint16_t> c_residual = fused_residual;
    std::vector<uint16_t> wrapper_x = fused_x;
    std::vector<uint16_t> wrapper_residual = fused_residual;
    void* fused_x_pointer = wrapper_x.data();
    void* fused_residual_pointer = wrapper_residual.data();
    void* fused_weight_pointer = zero_weight.data();
    uint16_t* fused_c_x_pointer = c_x.data();
    uint16_t* fused_c_residual_pointer = c_residual.data();
    void* fused_args[] = {&fused_x_pointer, &fused_residual_pointer,
                          &fused_weight_pointer};
    auto run_fused = [&] {
      fused(fused_x.data(), fused_residual.data(), zero_weight.data(), 0, 0, 0,
            1, 1, 1);
    };
    auto run_fused_wrapper = [&] {
      fused_wrapper.launch_with_signature(1, 1, 1, 1, nullptr, fused_args,
                                          "*bf16,*bf16,*bf16", 3);
    };
    auto run_fused_c = [&] {
      neon_vllm_fused_rms(fused_c_x_pointer, fused_c_residual_pointer,
                          zero_weight.data(), size, eps);
    };
    run_fused();
    run_fused_c();
    run_fused_wrapper();
    const size_t fused_mismatch =
        mismatches(fused_x, c_x) + mismatches(fused_residual, c_residual) +
        mismatches(fused_x, wrapper_x) +
        mismatches(fused_residual, wrapper_residual);
    fused_x_pointer = fused_x.data();
    fused_residual_pointer = fused_residual.data();
    fused_c_x_pointer = fused_x.data();
    fused_c_residual_pointer = fused_residual.data();
    const TripleTiming fused_timing = median_three_us(
        run_fused, run_fused_wrapper, run_fused_c, 20000);

    constexpr int q_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int position = 17;
    std::vector<uint16_t> q(q_heads * head_dim), k(kv_heads * head_dim);
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = to_bf16(std::sin(i * 0.019f));
    }
    for (size_t i = 0; i < k.size(); ++i) {
      k[i] = to_bf16(std::cos(i * 0.023f));
    }
    std::vector<int64_t> positions(1, position);
    std::vector<uint16_t> cache(32 * head_dim, to_bf16(0.0f));
    for (int i = 0; i < head_dim / 2; ++i) {
      cache[position * head_dim + i] = to_bf16(1.0f);
    }
    std::vector<uint16_t> c_q = q, c_k = k;
    std::vector<uint16_t> wrapper_q = q, wrapper_k = k;
    SharedObject rope_lib(argv[3]);
    FourPointerKernel rope =
        rope_lib.symbol<FourPointerKernel>("_rope_qk_aot_kernel");
    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> rope_wrapper(
        dirname(argv[3]), "_rope_qk_aot_kernel");
    void* rope_q_pointer = wrapper_q.data();
    void* rope_k_pointer = wrapper_k.data();
    void* rope_position_pointer = positions.data();
    void* rope_cache_pointer = cache.data();
    uint16_t* rope_c_q_pointer = c_q.data();
    uint16_t* rope_c_k_pointer = c_k.data();
    void* rope_args[] = {&rope_q_pointer, &rope_k_pointer,
                         &rope_position_pointer, &rope_cache_pointer};
    auto run_rope = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid) {
        rope(q.data(), k.data(), positions.data(), cache.data(), pid, 0, 0,
             q_heads + kv_heads, 1, 1);
      }
    };
    auto run_rope_wrapper = [&] {
      rope_wrapper.launch_with_signature(24, 1, 1, 1, nullptr, rope_args,
                                         "*bf16,*bf16,*i64,*bf16", 4);
    };
    auto run_rope_c = [&] {
      neon_rope(rope_c_q_pointer, rope_c_k_pointer, positions.data(),
                cache.data(), q_heads, kv_heads, head_dim);
    };
    run_rope();
    run_rope_c();
    run_rope_wrapper();
    const size_t rope_mismatch =
        mismatches(q, c_q) + mismatches(k, c_k) +
        mismatches(q, wrapper_q) + mismatches(k, wrapper_k);
    rope_q_pointer = q.data();
    rope_k_pointer = k.data();
    rope_c_q_pointer = q.data();
    rope_c_k_pointer = k.data();
    const TripleTiming rope_timing =
        median_three_us(run_rope, run_rope_wrapper, run_rope_c, 5000);

    constexpr int swiglu_size = 3072;
    std::vector<uint16_t> gate_up(swiglu_size * 2), swiglu_out(swiglu_size),
        swiglu_c_out(swiglu_size), swiglu_wrapper_out(swiglu_size);
    for (int i = 0; i < swiglu_size; ++i) {
      gate_up[i] = to_bf16(std::sin(i * 0.011f) * 3.0f);
      gate_up[swiglu_size + i] = to_bf16(std::cos(i * 0.007f));
    }
    SharedObject swiglu_lib(argv[4]);
    TwoPointerKernel swiglu = swiglu_lib.symbol<TwoPointerKernel>(
        "_bf16_swiglu_inline_exp_kernel");
    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> swiglu_wrapper(
        dirname(argv[4]), "_bf16_swiglu_inline_exp_kernel");
    void* swiglu_input_pointer = gate_up.data();
    void* swiglu_output_pointer = swiglu_wrapper_out.data();
    uint16_t* swiglu_c_out_pointer = swiglu_c_out.data();
    void* swiglu_args[] = {&swiglu_input_pointer, &swiglu_output_pointer};
    auto run_swiglu = [&] {
      swiglu(gate_up.data(), swiglu_out.data(), 0, 0, 0, 1, 1, 1);
    };
    auto run_swiglu_wrapper = [&] {
      swiglu_wrapper.launch_with_signature(1, 1, 1, 1, nullptr, swiglu_args,
                                           "*bf16,*bf16", 2);
    };
    auto run_swiglu_c = [&] {
      neon_swiglu(gate_up.data(), swiglu_c_out_pointer, swiglu_size);
    };
    run_swiglu();
    run_swiglu_c();
    run_swiglu_wrapper();
    const size_t swiglu_mismatch = mismatches(swiglu_out, swiglu_c_out) +
                                   mismatches(swiglu_out,
                                              swiglu_wrapper_out);
    swiglu_output_pointer = swiglu_out.data();
    swiglu_c_out_pointer = swiglu_out.data();
    const TripleTiming swiglu_timing = median_three_us(
        run_swiglu, run_swiglu_wrapper, run_swiglu_c, 3000);

    std::cout << "PASS current ordinary-Triton AOT vs equivalent ACLE C\n";
    print_result("rms", rms_timing.first, rms_timing.second, rms_timing.third,
                 rms_mismatch, size * 2);
    print_result("fused_rms", fused_timing.first, fused_timing.second,
                 fused_timing.third,
                 fused_mismatch, size * 4);
    print_result("rope", rope_timing.first, rope_timing.second,
                 rope_timing.third, rope_mismatch,
                 (q.size() + k.size()) * 2);
    print_result("swiglu", swiglu_timing.first, swiglu_timing.second,
                 swiglu_timing.third, swiglu_mismatch, swiglu_size * 2);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
