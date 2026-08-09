#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using ThreePointerKernel = void (*)(void*, void*, void*, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t, uint32_t);
using FourPointerKernel = void (*)(void*, void*, void*, void*, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t);
using W8Kernel = void (*)(void*, void*, void*, void*, int32_t, int32_t,
                          uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                          uint32_t);

class Library {
 public:
  explicit Library(const char* path) : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle_) throw std::runtime_error(dlerror());
  }
  ~Library() { dlclose(handle_); }
  template <typename Function>
  Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle_, name);
    if (const char* error = dlerror()) throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }

 private:
  void* handle_;
};

class Random {
 public:
  explicit Random(uint64_t seed) : state_(seed) {}
  uint32_t next() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return static_cast<uint32_t>((state_ * 2685821657736338717ULL) >> 32);
  }
  float uniform(float low, float high) {
    const float unit = static_cast<float>(next() >> 8) * (1.0f / 16777216.0f);
    return low + (high - low) * unit;
  }
  int integer(int low, int high) {
    return low + static_cast<int>(next() % static_cast<uint32_t>(high - low + 1));
  }

 private:
  uint64_t state_;
};

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float from_bf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

template <typename Value>
void store(std::vector<uint8_t>& buffer, size_t offset, Value value) {
  std::memcpy(buffer.data() + offset, &value, sizeof(value));
}

struct Difference {
  uint64_t exact_mismatches = 0;
  uint64_t tolerance_mismatches = 0;
  uint32_t max_bf16_ulp = 0;
  double max_abs = 0.0;
};

void merge_bf16(Difference& result, const std::vector<uint16_t>& reference,
                 const std::vector<uint16_t>& candidate) {
  if (reference.size() != candidate.size()) throw std::runtime_error("size mismatch");
  for (size_t i = 0; i < reference.size(); ++i) {
    const uint32_t ulp = static_cast<uint32_t>(std::abs(
        static_cast<int>(reference[i]) - static_cast<int>(candidate[i])));
    const double absolute =
        std::abs(static_cast<double>(from_bf16(reference[i])) -
                 static_cast<double>(from_bf16(candidate[i])));
    result.exact_mismatches += reference[i] != candidate[i];
    result.tolerance_mismatches += ulp > 2 && absolute > 2.0e-3;
    result.max_bf16_ulp = std::max(result.max_bf16_ulp, ulp);
    result.max_abs = std::max(result.max_abs, absolute);
  }
}

void merge_f32(Difference& result, const std::vector<float>& reference,
               const std::vector<float>& candidate, size_t begin,
               size_t end) {
  for (size_t i = begin; i < end; ++i) {
    const double absolute = std::abs(static_cast<double>(reference[i]) -
                                     static_cast<double>(candidate[i]));
    const double tolerance =
        2.0e-6 + 2.0e-6 * std::abs(static_cast<double>(reference[i]));
    result.exact_mismatches +=
        std::bit_cast<uint32_t>(reference[i]) != std::bit_cast<uint32_t>(candidate[i]);
    result.tolerance_mismatches += absolute > tolerance;
    result.max_abs = std::max(result.max_abs, absolute);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: " << argv[0]
              << " PORTABLE_C_SO ACLE_C_SO TRITON_RMS_SO TRITON_ROPE_SO"
                 " TRITON_W8_SO\n";
    return 2;
  }
  try {
    Library c_library(argv[1]);
    Library acle_library(argv[2]);
    Library rms_library(argv[3]);
    Library rope_library(argv[4]);
    Library w8_library(argv[5]);
    const auto c_rms =
        c_library.symbol<ThreePointerKernel>("_rms_same_backend_c");
    const auto acle_rms =
        acle_library.symbol<ThreePointerKernel>("_rms_same_backend_acle");
    const auto triton_rms = rms_library.symbol<ThreePointerKernel>(
        "_rms_same_backend_triton");
    const auto c_rope =
        c_library.symbol<FourPointerKernel>("_rope_same_backend_c");
    const auto acle_rope =
        acle_library.symbol<FourPointerKernel>("_rope_same_backend_acle");
    const auto triton_rope = rope_library.symbol<FourPointerKernel>(
        "_rope_same_backend_triton");
    const auto c_w8 = c_library.symbol<W8Kernel>("_w8_same_backend_c");
    const auto acle_w8 =
        acle_library.symbol<W8Kernel>("_w8_same_backend_acle");
    const auto triton_w8 = w8_library.symbol<W8Kernel>(
        "_kai_w8_layout_pointer_kernel");

    Random random(0x68f1d29ba457c301ULL);
    Difference rms_c_difference, rms_acle_difference;
    constexpr int rms_size = 1024;
    std::vector<uint16_t> input(rms_size), weight(rms_size), c_rms_out(rms_size),
        acle_rms_out(rms_size), triton_rms_out(rms_size);
    for (int test = 0; test < 64; ++test) {
      for (int i = 0; i < rms_size; ++i) {
        float x = random.uniform(-4.0f, 4.0f);
        float w = random.uniform(-2.0f, 2.0f);
        if ((i + test) % 127 == 0) x = 0.0f;
        if ((i + test) % 251 == 0) w = 0.0f;
        input[i] = to_bf16(x);
        weight[i] = to_bf16(w);
      }
      c_rms(input.data(), weight.data(), c_rms_out.data(), 0, 0, 0, 1, 1, 1);
      acle_rms(input.data(), weight.data(), acle_rms_out.data(), 0, 0, 0, 1,
               1, 1);
      triton_rms(input.data(), weight.data(), triton_rms_out.data(), 0, 0, 0,
                 1, 1, 1);
      merge_bf16(rms_c_difference, c_rms_out, triton_rms_out);
      merge_bf16(rms_acle_difference, acle_rms_out, triton_rms_out);
    }

    constexpr int q_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    const size_t q_size = q_heads * head_dim;
    const size_t k_size = kv_heads * head_dim;
    std::vector<uint16_t> q(q_size), k(k_size), cache(32 * head_dim);
    std::vector<uint16_t> c_q, c_k, acle_q, acle_k, triton_q, triton_k;
    std::vector<int64_t> positions(1);
    Difference rope_c_difference, rope_acle_difference;
    for (int test = 0; test < 32; ++test) {
      for (uint16_t& value : q) value = to_bf16(random.uniform(-3.0f, 3.0f));
      for (uint16_t& value : k) value = to_bf16(random.uniform(-3.0f, 3.0f));
      for (uint16_t& value : cache)
        value = to_bf16(random.uniform(-1.0f, 1.0f));
      positions[0] = random.integer(0, 31);
      c_q = q;
      c_k = k;
      acle_q = q;
      acle_k = k;
      triton_q = q;
      triton_k = k;
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid) {
        c_rope(c_q.data(), c_k.data(), positions.data(), cache.data(), pid, 0,
               0, q_heads + kv_heads, 1, 1);
        acle_rope(acle_q.data(), acle_k.data(), positions.data(), cache.data(),
                  pid, 0, 0, q_heads + kv_heads, 1, 1);
        triton_rope(triton_q.data(), triton_k.data(), positions.data(),
                    cache.data(), pid, 0, 0, q_heads + kv_heads, 1, 1);
      }
      c_q.insert(c_q.end(), c_k.begin(), c_k.end());
      acle_q.insert(acle_q.end(), acle_k.begin(), acle_k.end());
      triton_q.insert(triton_q.end(), triton_k.begin(), triton_k.end());
      merge_bf16(rope_c_difference, c_q, triton_q);
      merge_bf16(rope_acle_difference, acle_q, triton_q);
    }

    constexpr int w8_k = 1024;
    constexpr int w8_n = 3072;
    constexpr int blocks = w8_n / 4;
    constexpr int rhs_stride = 4 * w8_k + 48;
    std::vector<uint8_t> lhs(w8_k + 8);
    std::vector<uint8_t> rhs(static_cast<size_t>(blocks) * rhs_stride);
    std::vector<float> c_w8_out(w8_n), acle_w8_out(w8_n), triton_w8_out(w8_n);
    Difference w8_c_difference, w8_acle_difference;
    for (int test = 0; test < 8; ++test) {
      for (int i = 0; i < w8_k; ++i)
        lhs[i] = static_cast<uint8_t>(random.integer(-127, 127));
      store<int32_t>(lhs, w8_k, random.integer(-12, 12));
      store<float>(lhs, w8_k + 4, random.uniform(0.001f, 0.02f));
      for (int block = 0; block < blocks; ++block) {
        const size_t base = static_cast<size_t>(block) * rhs_stride;
        for (int i = 0; i < 4 * w8_k; ++i)
          rhs[base + i] = static_cast<uint8_t>(random.integer(-127, 127));
        for (int lane = 0; lane < 4; ++lane) {
          store<int32_t>(rhs, base + 4 * w8_k + lane * 4,
                         random.integer(-2048, 2048));
          store<float>(rhs, base + 4 * w8_k + 16 + lane * 4,
                       random.uniform(0.0005f, 0.003f));
          store<float>(rhs, base + 4 * w8_k + 32 + lane * 4,
                       random.uniform(-0.2f, 0.2f));
        }
      }
      const float clamp[2] = {
          test & 1 ? -0.75f : -1.0e30f,
          test & 1 ? 0.75f : 1.0e30f,
      };
      std::fill(c_w8_out.begin(), c_w8_out.end(),
                std::numeric_limits<float>::quiet_NaN());
      std::fill(acle_w8_out.begin(), acle_w8_out.end(),
                std::numeric_limits<float>::quiet_NaN());
      std::fill(triton_w8_out.begin(), triton_w8_out.end(),
                std::numeric_limits<float>::quiet_NaN());
      const int begin = test == 7 ? 16 : 0;
      const int end = test == 7 ? 32 : blocks;
      c_w8(lhs.data(), rhs.data(), const_cast<float*>(clamp), c_w8_out.data(),
           begin, end, 0, 0, 0, 1, 1, 1);
      acle_w8(lhs.data(), rhs.data(), const_cast<float*>(clamp),
              acle_w8_out.data(), begin, end, 0, 0, 0, 1, 1, 1);
      triton_w8(lhs.data(), rhs.data(), const_cast<float*>(clamp),
                triton_w8_out.data(), begin, end, 0, 0, 0, 1, 1, 1);
      const size_t output_begin = static_cast<size_t>(begin) * 4;
      const size_t output_end = static_cast<size_t>(end) * 4;
      merge_f32(w8_c_difference, c_w8_out, triton_w8_out, output_begin,
                output_end);
      merge_f32(w8_acle_difference, acle_w8_out, triton_w8_out, output_begin,
                output_end);
      for (size_t i = 0; i < c_w8_out.size(); ++i) {
        if (i >= output_begin && i < output_end) continue;
        if (!std::isnan(c_w8_out[i]) || !std::isnan(acle_w8_out[i]) ||
            !std::isnan(triton_w8_out[i]))
          throw std::runtime_error("W8 range wrote outside requested blocks");
      }
    }

    std::cout << "PASS randomized differential audit\n"
              << "rms_cases=64 portable_exact_mismatch="
              << rms_c_difference.exact_mismatches
              << " acle_exact_mismatch=" << rms_acle_difference.exact_mismatches
              << " portable_max_ulp=" << rms_c_difference.max_bf16_ulp
              << " acle_max_ulp=" << rms_acle_difference.max_bf16_ulp << '\n'
              << "rope_cases=32 portable_exact_mismatch="
              << rope_c_difference.exact_mismatches
              << " acle_exact_mismatch=" << rope_acle_difference.exact_mismatches
              << " portable_max_ulp=" << rope_c_difference.max_bf16_ulp
              << " acle_max_ulp=" << rope_acle_difference.max_bf16_ulp << '\n'
              << "w8_cases=8 portable_exact_mismatch="
              << w8_c_difference.exact_mismatches
              << " acle_exact_mismatch=" << w8_acle_difference.exact_mismatches
              << " portable_max_abs=" << w8_c_difference.max_abs
              << " acle_max_abs=" << w8_acle_difference.max_abs << '\n';
    if (rms_c_difference.tolerance_mismatches ||
        rms_acle_difference.tolerance_mismatches ||
        rope_c_difference.tolerance_mismatches ||
        rope_acle_difference.tolerance_mismatches ||
        w8_c_difference.tolerance_mismatches ||
        w8_acle_difference.tolerance_mismatches)
      throw std::runtime_error("random differential tolerance failure");
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
