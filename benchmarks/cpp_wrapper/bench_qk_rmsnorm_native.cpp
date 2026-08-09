#include <arm_neon.h>
#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using TritonKernel = void (*)(void *, void *, void *, int32_t, float,
                              uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t);

struct SharedObject {
  explicit SharedObject(const std::string &path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(dlerror());
  }
  ~SharedObject() {
    if (handle)
      dlclose(handle);
  }
  template <typename Function> Function symbol(const char *name) {
    dlerror();
    void *address = dlsym(handle, name);
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }
  void *handle = nullptr;
};

struct F32x8 {
  float32x4_t lo;
  float32x4_t hi;
};

inline F32x8 load_bf16(const uint16_t *pointer) {
  const bfloat16x8_t packed =
      vld1q_bf16(reinterpret_cast<const bfloat16_t *>(pointer));
  return {vcvt_f32_bf16(vget_low_bf16(packed)),
          vcvt_f32_bf16(vget_high_bf16(packed))};
}

inline void store_bf16(uint16_t *pointer, F32x8 value) {
  const bfloat16x8_t packed =
      vcombine_bf16(vcvt_bf16_f32(value.lo), vcvt_bf16_f32(value.hi));
  vst1q_bf16(reinterpret_cast<bfloat16_t *>(pointer), packed);
}

inline float tile_square_sum(F32x8 first, F32x8 second) {
  float32x4_t sum = vmulq_f32(first.lo, first.lo);
  sum = vfmaq_f32(sum, first.hi, first.hi);
  sum = vfmaq_f32(sum, second.lo, second.lo);
  sum = vfmaq_f32(sum, second.hi, second.hi);
  return vaddvq_f32(sum);
}

inline void acle_rms_row(const uint16_t *input, const uint16_t *weight,
                         uint16_t *output, int32_t head_dim, float eps) {
  float square_sum = 0.0f;
  for (int32_t offset = 0; offset < head_dim; offset += 16) {
    square_sum += tile_square_sum(load_bf16(input + offset),
                                  load_bf16(input + offset + 8));
  }
  const float rrms = 1.0f / std::sqrt(square_sum / head_dim + eps);
  const float32x4_t scale = vdupq_n_f32(rrms);
  for (int32_t offset = 0; offset < head_dim; offset += 8) {
    F32x8 value = load_bf16(input + offset);
    const F32x8 row_weight = load_bf16(weight + offset);
    value.lo =
        vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(value.lo, scale)));
    value.hi =
        vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(value.hi, scale)));
    store_bf16(output + offset,
               {vmulq_f32(value.lo, row_weight.lo),
                vmulq_f32(value.hi, row_weight.hi)});
  }
}

void acle_qk_rmsnorm(const uint16_t *input, const uint16_t *row_weight,
                     uint16_t *output, int32_t rows, int32_t head_dim,
                     float eps) {
  for (int32_t row = 0; row < rows; ++row) {
    acle_rms_row(input + static_cast<size_t>(row) * head_dim,
                 row_weight + static_cast<size_t>(row) * head_dim,
                 output + static_cast<size_t>(row) * head_dim, head_dim, eps);
  }
}

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

template <typename Function>
double batch_us(Function &&function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration)
    function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}

struct PairedTiming {
  double triton;
  double acle;
};

template <typename Triton, typename Acle>
PairedTiming paired_median_us(Triton &&triton, Acle &&acle, int warmup,
                              int iterations, int batches) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    triton();
    acle();
  }
  std::vector<double> triton_samples;
  std::vector<double> acle_samples;
  for (int batch = 0; batch < batches; ++batch) {
    if ((batch & 1) == 0) {
      triton_samples.push_back(batch_us(triton, iterations));
      acle_samples.push_back(batch_us(acle, iterations));
    } else {
      acle_samples.push_back(batch_us(acle, iterations));
      triton_samples.push_back(batch_us(triton, iterations));
    }
  }
  std::sort(triton_samples.begin(), triton_samples.end());
  std::sort(acle_samples.begin(), acle_samples.end());
  return {triton_samples[triton_samples.size() / 2],
          acle_samples[acle_samples.size() / 2]};
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 7 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " TRITON_SO Q_HEADS KV_HEADS HEAD_DIM ITERATIONS BATCHES"
                 " [EPS]\n";
    return 2;
  }
  try {
    const int q_heads = std::stoi(argv[2]);
    const int kv_heads = std::stoi(argv[3]);
    const int head_dim = std::stoi(argv[4]);
    const int iterations = std::stoi(argv[5]);
    const int batches = std::stoi(argv[6]);
    const float eps = argc == 8 ? std::stof(argv[7]) : 1.0e-6f;
    const int rows = q_heads + kv_heads;
    if (q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || head_dim % 16 ||
        iterations <= 0 || batches <= 0)
      throw std::runtime_error("invalid shape or repetition count");

    std::vector<uint16_t> input(static_cast<size_t>(rows) * head_dim);
    std::vector<uint16_t> weight(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
      input[index] = to_bf16(std::sin(index * 0.017f) * 1.3f);
      weight[index] = to_bf16(std::cos(index * 0.013f) * 0.4f + 0.8f);
    }
    std::vector<uint16_t> triton_output(input.size());
    std::vector<uint16_t> acle_output(input.size());
    std::vector<uint16_t> timed_output(input.size());

    SharedObject library(argv[1]);
    TritonKernel triton = library.symbol<TritonKernel>(
        "_qk_rms_norm_contiguous_kernel");
    auto call_triton_into = [&](uint16_t *output) {
      for (uint32_t pid = 0; pid < static_cast<uint32_t>(rows); ++pid) {
        triton(input.data(), weight.data(), output, head_dim, eps, pid, 0, 0,
               rows, 1, 1);
      }
    };
    auto call_acle_into = [&](uint16_t *output) {
      acle_qk_rmsnorm(input.data(), weight.data(), output, rows, head_dim,
                      eps);
    };
    const std::vector<uint16_t> timed_input = input;
    const std::vector<uint16_t> timed_weight = weight;
    constexpr int accuracy_cases = 9;
    size_t mismatches = 0;
    int max_bf16_ulp = 0;
    uint32_t random_state = 0x4b1d1237U;
    for (int accuracy_case = 0; accuracy_case < accuracy_cases;
         ++accuracy_case) {
      if (accuracy_case != 0) {
        for (size_t index = 0; index < input.size(); ++index) {
          random_state = random_state * 1664525U + 1013904223U;
          const float input_unit =
              static_cast<float>((random_state >> 8U) & 0xffffU) / 65535.0f;
          random_state = random_state * 1664525U + 1013904223U;
          const float weight_unit =
              static_cast<float>((random_state >> 8U) & 0xffffU) / 65535.0f;
          input[index] = to_bf16((input_unit * 2.0f - 1.0f) * 2.0f);
          weight[index] = to_bf16(0.5f + weight_unit);
        }
      }
      call_triton_into(triton_output.data());
      call_acle_into(acle_output.data());
      for (size_t index = 0; index < triton_output.size(); ++index) {
        const int ulp = std::abs(static_cast<int>(triton_output[index]) -
                                 static_cast<int>(acle_output[index]));
        mismatches += ulp != 0;
        max_bf16_ulp = std::max(max_bf16_ulp, ulp);
      }
    }
    if (mismatches)
      throw std::runtime_error("Triton/ACLE QK RMSNorm is not bit-exact");
    input = timed_input;
    weight = timed_weight;

    auto run_triton = [&] { call_triton_into(timed_output.data()); };
    auto run_acle = [&] { call_acle_into(timed_output.data()); };
    const auto timing =
        paired_median_us(run_triton, run_acle, 100, iterations, batches);
    std::cout << "PASS fused QK RMSNorm Hq=" << q_heads
              << " Hkv=" << kv_heads << " D=" << head_dim << '\n'
              << "triton_direct_us=" << timing.triton << '\n'
              << "acle_fused_us=" << timing.acle << '\n'
              << "triton_over_acle=" << timing.triton / timing.acle << "x\n"
              << "mismatches=" << mismatches << '/'
              << triton_output.size() * accuracy_cases
              << '\n'
              << "max_bf16_ulp=" << max_bf16_ulp << '\n'
              << "accuracy_cases=" << accuracy_cases << '\n'
              << "allocation_excluded=true\n"
              << "python_excluded=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
