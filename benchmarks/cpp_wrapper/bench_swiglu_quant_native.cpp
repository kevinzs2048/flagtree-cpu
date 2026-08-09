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

namespace {

using TritonKernel = void (*)(void *, void *, void *, void *, void *,
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

inline float32x4_t load_bf16x4(const uint16_t *pointer) {
  return vcvt_f32_bf16(
      vld1_bf16(reinterpret_cast<const bfloat16_t *>(pointer)));
}

inline bfloat16x4_t round_bf16(float32x4_t value) {
  return vcvt_bf16_f32(value);
}

// Keep four independent exp chains live at once.  This mirrors the scheduling
// opportunity in the 16-element Triton specialization instead of forcing a C
// compiler to finish one FDIV-dependent lane group before starting the next.
inline void exp_u10_inline_x4(const float32x4_t (&value)[4],
                             float32x4_t (&result)[4]) {
  int32x4_t exponent[4];
  float32x4_t reduced[4];
  float32x4_t polynomial[4];
  for (int lane = 0; lane < 4; ++lane) {
    exponent[lane] =
        vcvtnq_s32_f32(vmulq_n_f32(value[lane], 1.4426950408889634f));
    const float32x4_t exponent_f = vcvtq_f32_s32(exponent[lane]);
    reduced[lane] =
        vfmaq_n_f32(value[lane], exponent_f, -0.693145751953125f);
    reduced[lane] = vfmaq_n_f32(reduced[lane], exponent_f,
                                -1.428606765330187e-6f);
    polynomial[lane] = vdupq_n_f32(0.00019852761761285365f);
  }
  for (int stage = 0; stage < 5; ++stage) {
    constexpr float coefficients[5] = {
        0.0013930435525253415f, 0.008333360776305199f,
        0.041666485369205475f, 0.1666666716337204f, 0.5f};
    for (int lane = 0; lane < 4; ++lane)
      polynomial[lane] = vfmaq_f32(vdupq_n_f32(coefficients[stage]),
                                   polynomial[lane], reduced[lane]);
  }
  for (int lane = 0; lane < 4; ++lane) {
    result[lane] = vfmaq_f32(reduced[lane],
                             vmulq_f32(reduced[lane], reduced[lane]),
                             polynomial[lane]);
    result[lane] = vaddq_f32(vdupq_n_f32(1.0f), result[lane]);
    const int32x4_t half_exponent = vshrq_n_s32(exponent[lane], 1);
    const int32x4_t pow0_bits = vshlq_n_s32(
        vaddq_s32(half_exponent, vdupq_n_s32(127)), 23);
    const int32x4_t pow1_bits = vshlq_n_s32(
        vaddq_s32(vsubq_s32(exponent[lane], half_exponent),
                  vdupq_n_s32(127)),
        23);
    result[lane] =
        vmulq_f32(result[lane], vreinterpretq_f32_s32(pow0_bits));
    result[lane] =
        vmulq_f32(result[lane], vreinterpretq_f32_s32(pow1_bits));
    result[lane] = vbslq_f32(
        vcltq_f32(value[lane], vdupq_n_f32(-104.0f)), vdupq_n_f32(0.0f),
        result[lane]);
    result[lane] = vbslq_f32(
        vcgtq_f32(value[lane], vdupq_n_f32(100.0f)),
        vdupq_n_f32(std::numeric_limits<float>::infinity()), result[lane]);
  }
}

void acle_swiglu_quant(const uint16_t *gate, const uint16_t *up,
                       uint16_t *bf16, int8_t *quantized, float *scale,
                       int32_t elements) {
  uint16x4_t max_magnitude = vdup_n_u16(0);
  for (int32_t offset = 0; offset < elements; offset += 16) {
    float32x4_t gate_value[4];
    float32x4_t up_value[4];
    float32x4_t negative_gate[4];
    float32x4_t exponential[4];
    for (int lane = 0; lane < 4; ++lane) {
      gate_value[lane] = load_bf16x4(gate + offset + 4 * lane);
      up_value[lane] = load_bf16x4(up + offset + 4 * lane);
      negative_gate[lane] = vnegq_f32(gate_value[lane]);
    }
    exp_u10_inline_x4(negative_gate, exponential);
    for (int lane = 0; lane < 4; ++lane) {
      const float32x4_t denominator =
          vaddq_f32(vdupq_n_f32(1.0f), exponential[lane]);
      const bfloat16x4_t silu_bf16 =
          round_bf16(vdivq_f32(gate_value[lane], denominator));
      const bfloat16x4_t value_bf16 =
          round_bf16(vmulq_f32(vcvt_f32_bf16(silu_bf16), up_value[lane]));
      vst1_bf16(reinterpret_cast<bfloat16_t *>(bf16 + offset + 4 * lane),
                value_bf16);
      const uint16x4_t magnitude = vand_u16(
          vreinterpret_u16_bf16(value_bf16), vdup_n_u16(0x7fff));
      max_magnitude = vmax_u16(max_magnitude, magnitude);
    }
  }
  const uint16_t max_magnitude_bits = vmaxv_u16(max_magnitude);
  const float observed_absmax =
      std::bit_cast<float>(static_cast<uint32_t>(max_magnitude_bits) << 16U);
  const float absmax = std::max(observed_absmax, 1.0e-8f);
  scale[0] = absmax / 127.0f;
  const float inv_scale = 127.0f / absmax;
  for (int32_t offset = 0; offset < elements; offset += 16) {
    for (int32_t lane = 0; lane < 16; lane += 4) {
      const int32x4_t rounded = vcvtnq_s32_f32(
          vmulq_n_f32(load_bf16x4(bf16 + offset + lane), inv_scale));
      const int16x4_t narrowed16 = vqmovn_s32(rounded);
      const int8x8_t narrowed8 =
          vqmovn_s16(vcombine_s16(narrowed16, vdup_n_s16(0)));
      const int32_t packed =
          vget_lane_s32(vreinterpret_s32_s8(narrowed8), 0);
      std::memcpy(quantized + offset + lane, &packed, sizeof(packed));
    }
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

template <typename First, typename Second>
std::pair<double, double> paired_median_us(First &&first, Second &&second,
                                           int iterations, int batches) {
  for (int iteration = 0; iteration < 100; ++iteration) {
    first();
    second();
  }
  std::vector<double> first_samples;
  std::vector<double> second_samples;
  for (int batch = 0; batch < batches; ++batch) {
    if ((batch & 1) == 0) {
      first_samples.push_back(batch_us(first, iterations));
      second_samples.push_back(batch_us(second, iterations));
    } else {
      second_samples.push_back(batch_us(second, iterations));
      first_samples.push_back(batch_us(first, iterations));
    }
  }
  std::sort(first_samples.begin(), first_samples.end());
  std::sort(second_samples.begin(), second_samples.end());
  return {first_samples[first_samples.size() / 2],
          second_samples[second_samples.size() / 2]};
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) {
    std::cerr << "usage: " << argv[0]
              << " TRITON_SO ELEMENTS ITERATIONS [BATCHES]\n";
    return 2;
  }
  try {
    const int elements = std::stoi(argv[2]);
    const int iterations = std::stoi(argv[3]);
    const int batches = argc == 5 ? std::stoi(argv[4]) : 15;
    if (elements <= 0 || elements % 16 || iterations <= 0 || batches <= 0)
      throw std::runtime_error("invalid element or repetition count");
    std::vector<uint16_t> gate(elements), up(elements);
    for (int index = 0; index < elements; ++index) {
      gate[index] = to_bf16(std::sin(index * 0.017f) * 2.0f);
      up[index] = to_bf16(std::cos(index * 0.013f) * 1.5f);
    }
    std::vector<uint16_t> triton_bf16(elements), acle_bf16(elements),
        timed_bf16(elements);
    std::vector<int8_t> triton_q(elements), acle_q(elements), timed_q(elements);
    float triton_scale = 0.0f;
    float acle_scale = 0.0f;
    float timed_scale = 0.0f;
    SharedObject library(argv[1]);
    TritonKernel triton = library.symbol<TritonKernel>(
        "_swiglu_quantize_w8_rne_kernel");
    auto call_triton_into = [&](uint16_t *bf16, int8_t *q, float *scale) {
      triton(gate.data(), up.data(), bf16, q, scale, 0, 0, 0, 1, 1, 1);
    };
    auto call_acle_into = [&](uint16_t *bf16, int8_t *q, float *scale) {
      acle_swiglu_quant(gate.data(), up.data(), bf16, q, scale, elements);
    };
    const std::vector<uint16_t> timed_gate = gate;
    const std::vector<uint16_t> timed_up = up;
    constexpr int accuracy_cases = 16;
    size_t bf16_mismatch = 0;
    size_t q_mismatch = 0;
    bool scale_exact = true;
    uint32_t random_state = 0x9e3779b9U;
    for (int accuracy_case = 0; accuracy_case < accuracy_cases;
         ++accuracy_case) {
      if (accuracy_case != 0) {
        for (int index = 0; index < elements; ++index) {
          random_state = random_state * 1664525U + 1013904223U;
          const float gate_unit =
              static_cast<float>((random_state >> 8U) & 0xffffU) / 65535.0f;
          random_state = random_state * 1664525U + 1013904223U;
          const float up_unit =
              static_cast<float>((random_state >> 8U) & 0xffffU) / 65535.0f;
          gate[index] = to_bf16((gate_unit * 2.0f - 1.0f) * 8.0f);
          up[index] = to_bf16((up_unit * 2.0f - 1.0f) * 4.0f);
        }
      }
      call_triton_into(triton_bf16.data(), triton_q.data(), &triton_scale);
      call_acle_into(acle_bf16.data(), acle_q.data(), &acle_scale);
      for (int index = 0; index < elements; ++index) {
        bf16_mismatch += triton_bf16[index] != acle_bf16[index];
        q_mismatch += triton_q[index] != acle_q[index];
      }
      scale_exact &= std::bit_cast<uint32_t>(triton_scale) ==
                     std::bit_cast<uint32_t>(acle_scale);
    }
    if (bf16_mismatch || q_mismatch || !scale_exact)
      throw std::runtime_error("Triton/ACLE SwiGLU quant output differs");
    gate = timed_gate;
    up = timed_up;
    auto run_triton = [&] {
      call_triton_into(timed_bf16.data(), timed_q.data(), &timed_scale);
    };
    auto run_acle = [&] {
      call_acle_into(timed_bf16.data(), timed_q.data(), &timed_scale);
    };
    const auto [triton_us, acle_us] =
        paired_median_us(run_triton, run_acle, iterations, batches);
    std::cout << "PASS fused SwiGLU/W8 quant N=" << elements << '\n'
              << "triton_direct_us=" << triton_us << '\n'
              << "acle_fused_us=" << acle_us << '\n'
              << "triton_over_acle=" << triton_us / acle_us << "x\n"
              << "bf16_mismatch=" << bf16_mismatch << '/'
              << elements * accuracy_cases << '\n'
              << "q_mismatch=" << q_mismatch << '/'
              << elements * accuracy_cases << '\n'
              << "scale_bit_exact=" << (scale_exact ? "true" : "false")
              << '\n'
              << "accuracy_cases=" << accuracy_cases << '\n'
              << "allocation_excluded=true\n"
              << "python_excluded=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
