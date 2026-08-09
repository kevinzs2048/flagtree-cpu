#include <arm_neon.h>
#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {
using TritonKernel = void (*)(void*, void*, int32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t);

struct SharedObject {
  explicit SharedObject(const std::string& path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(dlerror());
  }
  ~SharedObject() {
    if (handle) dlclose(handle);
  }
  template <typename Function>
  Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle, name);
    if (const char* error = dlerror()) throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }
  void* handle = nullptr;
};

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

template <typename Function>
double median_us(Function&& function, int warmup, int iterations, int batches) {
  for (int i = 0; i < warmup; ++i) function();
  std::vector<double> samples;
  for (int batch = 0; batch < batches; ++batch) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) function();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count() /
        iterations);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::string dirname(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

float32x4_t bf16x4_to_f32(uint16x4_t value) {
  return vreinterpretq_f32_u32(vshll_n_u16(value, 16));
}

__attribute__((noinline)) int64_t neon_argmax_finite(const uint16_t* input,
                                                     int32_t size) {
  float32x4_t max0 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  float32x4_t max1 = max0;
  float32x4_t max2 = max0;
  float32x4_t max3 = max0;
  int32x4_t index0 = vdupq_n_s32(0);
  int32x4_t index1 = index0;
  int32x4_t index2 = index0;
  int32x4_t index3 = index0;
  int32x4_t current0 = {0, 1, 2, 3};
  int32x4_t current1 = {4, 5, 6, 7};
  int32x4_t current2 = {8, 9, 10, 11};
  int32x4_t current3 = {12, 13, 14, 15};
  const int32x4_t increment = vdupq_n_s32(16);

  int32_t offset = 0;
  for (; offset + 16 <= size; offset += 16) {
    const uint16x8_t packed0 = vld1q_u16(input + offset);
    const uint16x8_t packed1 = vld1q_u16(input + offset + 8);
    const float32x4_t value0 = bf16x4_to_f32(vget_low_u16(packed0));
    const float32x4_t value1 = bf16x4_to_f32(vget_high_u16(packed0));
    const float32x4_t value2 = bf16x4_to_f32(vget_low_u16(packed1));
    const float32x4_t value3 = bf16x4_to_f32(vget_high_u16(packed1));
    const uint32x4_t update0 = vcgtq_f32(value0, max0);
    const uint32x4_t update1 = vcgtq_f32(value1, max1);
    const uint32x4_t update2 = vcgtq_f32(value2, max2);
    const uint32x4_t update3 = vcgtq_f32(value3, max3);
    max0 = vbslq_f32(update0, value0, max0);
    max1 = vbslq_f32(update1, value1, max1);
    max2 = vbslq_f32(update2, value2, max2);
    max3 = vbslq_f32(update3, value3, max3);
    index0 = vbslq_s32(update0, current0, index0);
    index1 = vbslq_s32(update1, current1, index1);
    index2 = vbslq_s32(update2, current2, index2);
    index3 = vbslq_s32(update3, current3, index3);
    current0 = vaddq_s32(current0, increment);
    current1 = vaddq_s32(current1, increment);
    current2 = vaddq_s32(current2, increment);
    current3 = vaddq_s32(current3, increment);
  }

  alignas(16) float maxima[16];
  alignas(16) int32_t indices[16];
  vst1q_f32(maxima, max0);
  vst1q_f32(maxima + 4, max1);
  vst1q_f32(maxima + 8, max2);
  vst1q_f32(maxima + 12, max3);
  vst1q_s32(indices, index0);
  vst1q_s32(indices + 4, index1);
  vst1q_s32(indices + 8, index2);
  vst1q_s32(indices + 12, index3);
  float best = -std::numeric_limits<float>::infinity();
  int32_t best_index = 0;
  for (int lane = 0; lane < 16; ++lane) {
    if (maxima[lane] > best ||
        (maxima[lane] == best && indices[lane] < best_index)) {
      best = maxima[lane];
      best_index = indices[lane];
    }
  }
  for (; offset < size; ++offset) {
    const float value = std::bit_cast<float>(
        static_cast<uint32_t>(input[offset]) << 16U);
    if (value > best) {
      best = value;
      best_index = offset;
    }
  }
  return best_index;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::cerr << "usage: " << argv[0]
              << " KERNEL_SO [ITERS=2000] [BATCHES=9]\n";
    return 2;
  }
  try {
    const std::string kernel_path = argv[1];
    const int iterations = argc >= 3 ? std::stoi(argv[2]) : 2000;
    const int batches = argc >= 4 ? std::stoi(argv[3]) : 9;
    constexpr int32_t size = 151936;
    std::vector<uint16_t> input(size);
    for (int32_t i = 0; i < size; ++i) {
      input[i] = to_bf16(std::sin(i * 0.017f));
    }
    input[17] = input[65553] = to_bf16(100.0f);

    SharedObject library(kernel_path);
    TritonKernel direct =
        library.symbol<TritonKernel>("argmax_vocab_rolled_kernel");
    int64_t direct_output = -1;
    int64_t wrapper_output = -1;
    auto run_direct = [&] {
      direct(input.data(), &direct_output, size, 0, 0, 0, 1, 1, 1);
    };

    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> wrapper(
        dirname(kernel_path), "argmax_vocab_rolled_kernel");
    void* input_ptr = input.data();
    void* output_ptr = &wrapper_output;
    int32_t size_arg = size;
    void* args[] = {&input_ptr, &output_ptr, &size_arg};
    auto run_wrapper = [&] {
      wrapper.launch_with_signature(1, 1, 1, 1, nullptr, args,
                                    "*bf16,*i64,i32", 3);
    };

    volatile int64_t c_output = -1;
    auto run_c = [&] {
      asm volatile("" : : "r"(input.data()) : "memory");
      c_output = neon_argmax_finite(input.data(), size);
    };
    run_direct();
    run_wrapper();
    run_c();
    if (direct_output != 17 || wrapper_output != 17 || c_output != 17) {
      throw std::runtime_error("argmax correctness mismatch");
    }

    constexpr int warmup = 100;
    const double direct_us =
        median_us(run_direct, warmup, iterations, batches);
    const double wrapper_us =
        median_us(run_wrapper, warmup, iterations, batches);
    const double c_us = median_us(run_c, warmup, iterations, batches);
    std::cout << "PASS N=" << size << " BF16 finite logits\n"
              << "direct_aot_triton_us=" << direct_us << '\n'
              << "libtriton_jit_wrapper_us=" << wrapper_us << '\n'
              << "optimized_neon_c_us=" << c_us << '\n'
              << "wrapper_minus_direct_us=" << wrapper_us - direct_us << '\n'
              << "triton_over_c=" << direct_us / c_us << "x\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
