#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

template <typename Value>
class AlignedArray {
 public:
  explicit AlignedArray(size_t size) : size_(size) {
    const size_t bytes = size * sizeof(Value);
    const size_t allocation = (bytes + 4095U) & ~size_t{4095U};
    data_ = static_cast<Value*>(std::aligned_alloc(4096, allocation));
    if (!data_) throw std::bad_alloc();
  }
  ~AlignedArray() { std::free(data_); }
  AlignedArray(const AlignedArray&) = delete;
  AlignedArray& operator=(const AlignedArray&) = delete;
  Value* data() { return data_; }
  const Value* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  Value* data_ = nullptr;
  size_t size_ = 0;
};

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float from_bf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

template <typename Function>
double timed_us(Function&& function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}

struct PairTiming {
  double first;
  double second;
};

template <typename First, typename Second>
PairTiming paired_us(First&& first, Second&& second, int iterations,
                     int batches = 21) {
  for (int i = 0; i < iterations; ++i) {
    first();
    second();
  }
  std::vector<double> a, b;
  for (int batch = 0; batch < batches; ++batch) {
    if ((batch & 1) == 0) {
      a.push_back(timed_us(first, iterations));
      b.push_back(timed_us(second, iterations));
    } else {
      b.push_back(timed_us(second, iterations));
      a.push_back(timed_us(first, iterations));
    }
  }
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return {a[a.size() / 2], b[b.size() / 2]};
}

struct TripleTiming {
  double portable_c;
  double acle_c;
  double triton;
};

template <typename Portable, typename Acle, typename Triton>
TripleTiming triple_us(Portable&& portable, Acle&& acle, Triton&& triton,
                       int iterations, int batches = 30) {
  for (int i = 0; i < iterations; ++i) {
    portable();
    acle();
    triton();
  }
  std::vector<double> portable_samples, acle_samples, triton_samples;
  for (int batch = 0; batch < batches; ++batch) {
    switch (batch % 3) {
      case 0:
        portable_samples.push_back(timed_us(portable, iterations));
        acle_samples.push_back(timed_us(acle, iterations));
        triton_samples.push_back(timed_us(triton, iterations));
        break;
      case 1:
        acle_samples.push_back(timed_us(acle, iterations));
        triton_samples.push_back(timed_us(triton, iterations));
        portable_samples.push_back(timed_us(portable, iterations));
        break;
      default:
        triton_samples.push_back(timed_us(triton, iterations));
        portable_samples.push_back(timed_us(portable, iterations));
        acle_samples.push_back(timed_us(acle, iterations));
        break;
    }
  }
  std::sort(portable_samples.begin(), portable_samples.end());
  std::sort(acle_samples.begin(), acle_samples.end());
  std::sort(triton_samples.begin(), triton_samples.end());
  const size_t median = portable_samples.size() / 2;
  return {portable_samples[median], acle_samples[median],
          triton_samples[median]};
}

template <typename Portable, typename Acle, typename Triton>
TripleTiming frontend_us(Portable&& portable, Acle&& acle, Triton&& triton,
                         int iterations, int batches,
                         bool optimized_only) {
  if (!optimized_only)
    return triple_us(portable, acle, triton, iterations, batches);
  const auto [acle_us, triton_us] =
      paired_us(acle, triton, iterations, batches);
  return {std::numeric_limits<double>::quiet_NaN(), acle_us, triton_us};
}

template <typename Value>
void store(std::vector<uint8_t>& buffer, size_t offset, Value value) {
  std::memcpy(buffer.data() + offset, &value, sizeof(value));
}

struct Error {
  size_t mismatches = 0;
  int max_bf16_ulp = 0;
  double max_abs = 0.0;
};

Error compare_bf16(const std::vector<uint16_t>& a,
                   const std::vector<uint16_t>& b) {
  Error error;
  for (size_t i = 0; i < a.size(); ++i) {
    const int ulp = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    error.mismatches += ulp != 0;
    error.max_bf16_ulp = std::max(error.max_bf16_ulp, ulp);
    error.max_abs = std::max(
        error.max_abs,
        static_cast<double>(std::abs(from_bf16(a[i]) - from_bf16(b[i]))));
  }
  return error;
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
    const bool optimized_only =
        std::getenv("SAME_BACKEND_OPTIMIZED_ONLY") != nullptr;
    Library c_library(argv[1]);
    Library acle_library(argv[2]);
    Library rms_library(argv[3]);
    Library rope_library(argv[4]);
    Library w8_library(argv[5]);
    auto c_rms = c_library.symbol<ThreePointerKernel>("_rms_same_backend_c");
    auto acle_rms =
        acle_library.symbol<ThreePointerKernel>("_rms_same_backend_acle");
    auto triton_rms = rms_library.symbol<ThreePointerKernel>(
        "_rms_same_backend_triton");
    auto c_rope = c_library.symbol<FourPointerKernel>("_rope_same_backend_c");
    auto acle_rope =
        acle_library.symbol<FourPointerKernel>("_rope_same_backend_acle");
    auto triton_rope = rope_library.symbol<FourPointerKernel>(
        "_rope_same_backend_triton");
    auto c_w8 = c_library.symbol<W8Kernel>("_w8_same_backend_c");
    auto acle_w8 =
        acle_library.symbol<W8Kernel>("_w8_same_backend_acle");
    auto triton_w8 = w8_library.symbol<W8Kernel>(
        "_kai_w8_layout_pointer_kernel");

    constexpr int n = 1024;
    std::vector<uint16_t> x(n), weight(n), c_rms_out(n), acle_rms_out(n),
        triton_rms_out(n);
    for (int i = 0; i < n; ++i) {
      x[i] = to_bf16(std::sin(i * 0.017f));
      weight[i] = to_bf16(std::cos(i * 0.013f));
    }
    c_rms(x.data(), weight.data(), c_rms_out.data(), 0, 0, 0, 1, 1, 1);
    acle_rms(x.data(), weight.data(), acle_rms_out.data(), 0, 0, 0, 1, 1, 1);
    triton_rms(x.data(), weight.data(), triton_rms_out.data(), 0, 0, 0, 1, 1,
               1);
    const Error rms_error = compare_bf16(c_rms_out, triton_rms_out);
    const Error rms_acle_error = compare_bf16(acle_rms_out, triton_rms_out);
    AlignedArray<uint16_t> timed_x(n), timed_weight(n), timed_rms_out(n);
    std::copy(x.begin(), x.end(), timed_x.data());
    std::copy(weight.begin(), weight.end(), timed_weight.data());
    // All timed implementations use page-aligned independent allocations and
    // write the same address.  A separate sweep covers non-cache-line offsets.
    auto run_c_rms = [&] {
      c_rms(timed_x.data(), timed_weight.data(), timed_rms_out.data(), 0, 0, 0,
            1, 1, 1);
    };
    auto run_acle_rms = [&] {
      acle_rms(timed_x.data(), timed_weight.data(), timed_rms_out.data(), 0, 0,
               0, 1, 1, 1);
    };
    auto run_triton_rms = [&] {
      triton_rms(timed_x.data(), timed_weight.data(), timed_rms_out.data(), 0,
                 0, 0, 1, 1, 1);
    };
    const TripleTiming rms_timing =
        frontend_us(run_c_rms, run_acle_rms, run_triton_rms, 20000, 30,
                    optimized_only);

    constexpr int q_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int position = 17;
    std::vector<uint16_t> c_q(q_heads * head_dim), c_k(kv_heads * head_dim);
    for (size_t i = 0; i < c_q.size(); ++i)
      c_q[i] = to_bf16(std::sin(i * 0.019f));
    for (size_t i = 0; i < c_k.size(); ++i)
      c_k[i] = to_bf16(std::cos(i * 0.023f));
    std::vector<uint16_t> acle_q = c_q, acle_k = c_k;
    std::vector<uint16_t> triton_q = c_q, triton_k = c_k;
    std::vector<int64_t> positions(1, position);
    std::vector<uint16_t> cache(32 * head_dim);
    for (size_t i = 0; i < cache.size(); ++i)
      cache[i] = to_bf16(std::sin(i * 0.0013f) * 0.25f + 0.75f);
    auto call_c_rope = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
        c_rope(c_q.data(), c_k.data(), positions.data(), cache.data(), pid, 0,
               0, q_heads + kv_heads, 1, 1);
    };
    auto call_triton_rope = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
        triton_rope(triton_q.data(), triton_k.data(), positions.data(),
                    cache.data(), pid, 0, 0, q_heads + kv_heads, 1, 1);
    };
    auto call_acle_rope = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
        acle_rope(acle_q.data(), acle_k.data(), positions.data(), cache.data(),
                  pid, 0, 0, q_heads + kv_heads, 1, 1);
    };
    call_c_rope();
    call_acle_rope();
    call_triton_rope();
    std::vector<uint16_t> c_rope_joined = c_q;
    c_rope_joined.insert(c_rope_joined.end(), c_k.begin(), c_k.end());
    std::vector<uint16_t> triton_rope_joined = triton_q;
    triton_rope_joined.insert(triton_rope_joined.end(), triton_k.begin(),
                              triton_k.end());
    std::vector<uint16_t> acle_rope_joined = acle_q;
    acle_rope_joined.insert(acle_rope_joined.end(), acle_k.begin(),
                            acle_k.end());
    const Error rope_error = compare_bf16(c_rope_joined, triton_rope_joined);
    const Error rope_acle_error =
        compare_bf16(acle_rope_joined, triton_rope_joined);
    // Identity cache keeps the in-place timing data stable.
    std::fill(cache.begin(), cache.end(), to_bf16(0.0f));
    for (int i = 0; i < head_dim / 2; ++i)
      cache[position * head_dim + i] = to_bf16(1.0f);
    AlignedArray<uint16_t> timed_q(triton_q.size()), timed_k(triton_k.size()),
        timed_cache(cache.size());
    AlignedArray<int64_t> timed_positions(positions.size());
    std::copy(triton_q.begin(), triton_q.end(), timed_q.data());
    std::copy(triton_k.begin(), triton_k.end(), timed_k.data());
    std::copy(cache.begin(), cache.end(), timed_cache.data());
    std::copy(positions.begin(), positions.end(), timed_positions.data());
    auto timed_c_rope = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
        c_rope(timed_q.data(), timed_k.data(), timed_positions.data(),
               timed_cache.data(), pid, 0, 0, q_heads + kv_heads, 1, 1);
    };
    auto timed_acle_rope = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
        acle_rope(timed_q.data(), timed_k.data(), timed_positions.data(),
                  timed_cache.data(), pid, 0, 0, q_heads + kv_heads, 1, 1);
    };
    auto timed_triton_rope = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
        triton_rope(timed_q.data(), timed_k.data(), timed_positions.data(),
                    timed_cache.data(), pid, 0, 0, q_heads + kv_heads, 1, 1);
    };
    const TripleTiming rope_timing =
        frontend_us(timed_c_rope, timed_acle_rope, timed_triton_rope, 5000, 30,
                    optimized_only);

    constexpr int k_size = 1024;
    constexpr int w8_n = 3072;
    constexpr int blocks = w8_n / 4;
    constexpr int rhs_stride = 4 * k_size + 48;
    std::vector<uint8_t> lhs(k_size + 8);
    std::vector<uint8_t> rhs(static_cast<size_t>(blocks) * rhs_stride);
    for (int i = 0; i < k_size; ++i)
      lhs[i] = static_cast<uint8_t>((i * 17 + 5) % 255 - 127);
    store<int32_t>(lhs, k_size, -3);
    store<float>(lhs, k_size + 4, 0.013f);
    for (int block = 0; block < blocks; ++block) {
      const size_t base = static_cast<size_t>(block) * rhs_stride;
      for (int i = 0; i < 4 * k_size; ++i)
        rhs[base + i] =
            static_cast<uint8_t>((i * 29 + block * 11 + 7) % 255 - 127);
      for (int lane = 0; lane < 4; ++lane) {
        store<int32_t>(rhs, base + 4 * k_size + lane * 4, lane * 13 - 19);
        store<float>(rhs, base + 4 * k_size + 16 + lane * 4,
                     0.001f + lane * 0.00007f);
        store<float>(rhs, base + 4 * k_size + 32 + lane * 4,
                     lane * 0.0003f);
      }
    }
    const float clamp[2] = {-1.0e30f, 1.0e30f};
    std::vector<float> c_w8_out(w8_n), acle_w8_out(w8_n),
        triton_w8_out(w8_n);
    c_w8(lhs.data(), rhs.data(), const_cast<float*>(clamp), c_w8_out.data(), 0,
         blocks, 0, 0, 0, 1, 1, 1);
    acle_w8(lhs.data(), rhs.data(), const_cast<float*>(clamp),
            acle_w8_out.data(), 0, blocks, 0, 0, 0, 1, 1, 1);
    triton_w8(lhs.data(), rhs.data(), const_cast<float*>(clamp),
              triton_w8_out.data(), 0, blocks, 0, 0, 0, 1, 1, 1);
    Error w8_error;
    Error w8_acle_error;
    for (int i = 0; i < w8_n; ++i) {
      const double difference = std::abs(c_w8_out[i] - triton_w8_out[i]);
      w8_error.max_abs = std::max(w8_error.max_abs, difference);
      w8_error.mismatches +=
          difference > 2.0e-6 + 2.0e-6 * std::abs(c_w8_out[i]);
      const double acle_difference =
          std::abs(acle_w8_out[i] - triton_w8_out[i]);
      w8_acle_error.max_abs =
          std::max(w8_acle_error.max_abs, acle_difference);
      w8_acle_error.mismatches +=
          acle_difference > 2.0e-6 + 2.0e-6 * std::abs(acle_w8_out[i]);
    }
    if (w8_error.mismatches || w8_acle_error.mismatches)
      throw std::runtime_error("W8 frontend outputs differ");
    AlignedArray<uint8_t> timed_lhs(lhs.size()), timed_rhs(rhs.size());
    AlignedArray<float> timed_clamp(2), timed_w8_out(w8_n);
    std::copy(lhs.begin(), lhs.end(), timed_lhs.data());
    std::copy(rhs.begin(), rhs.end(), timed_rhs.data());
    timed_clamp.data()[0] = clamp[0];
    timed_clamp.data()[1] = clamp[1];
    auto run_c_w8 = [&] {
      c_w8(timed_lhs.data(), timed_rhs.data(), timed_clamp.data(),
           timed_w8_out.data(), 0, blocks, 0, 0, 0, 1, 1, 1);
    };
    auto run_acle_w8 = [&] {
      acle_w8(timed_lhs.data(), timed_rhs.data(), timed_clamp.data(),
              timed_w8_out.data(), 0, blocks, 0, 0, 0, 1, 1, 1);
    };
    auto run_triton_w8 = [&] {
      triton_w8(timed_lhs.data(), timed_rhs.data(), timed_clamp.data(),
                timed_w8_out.data(), 0, blocks, 0, 0, 0, 1, 1, 1);
    };
    const TripleTiming w8_timing =
        frontend_us(run_c_w8, run_acle_w8, run_triton_w8, 500, 30,
                    optimized_only);

    std::cout << "PASS portable-C vs ACLE-C vs Triton, common MLIR/LLVM23 backend\n"
              << "rms_portable_c_us=" << rms_timing.portable_c << '\n'
              << "rms_acle_c_us=" << rms_timing.acle_c << '\n'
              << "rms_triton_us=" << rms_timing.triton << '\n'
              << "rms_triton_over_portable="
              << rms_timing.triton / rms_timing.portable_c << "x\n"
              << "rms_triton_over_acle="
              << rms_timing.triton / rms_timing.acle_c << "x\n"
              << "rms_mismatch=" << rms_error.mismatches << '/' << n << '\n'
              << "rms_acle_mismatch=" << rms_acle_error.mismatches << '/' << n
              << '\n'
              << "rms_max_bf16_ulp=" << rms_error.max_bf16_ulp << '\n'
              << "rope_portable_c_us=" << rope_timing.portable_c << '\n'
              << "rope_acle_c_us=" << rope_timing.acle_c << '\n'
              << "rope_triton_us=" << rope_timing.triton << '\n'
              << "rope_triton_over_portable="
              << rope_timing.triton / rope_timing.portable_c << "x\n"
              << "rope_triton_over_acle="
              << rope_timing.triton / rope_timing.acle_c << "x\n"
              << "rope_mismatch=" << rope_error.mismatches << '/'
              << c_rope_joined.size() << '\n'
              << "rope_acle_mismatch=" << rope_acle_error.mismatches << '/'
              << acle_rope_joined.size() << '\n'
              << "rope_max_bf16_ulp=" << rope_error.max_bf16_ulp << '\n'
              << "w8_portable_c_us=" << w8_timing.portable_c << '\n'
              << "w8_acle_c_us=" << w8_timing.acle_c << '\n'
              << "w8_triton_us=" << w8_timing.triton << '\n'
              << "w8_triton_over_portable="
              << w8_timing.triton / w8_timing.portable_c << "x\n"
              << "w8_triton_over_acle="
              << w8_timing.triton / w8_timing.acle_c << "x\n"
              << "w8_max_abs=" << w8_error.max_abs << '\n'
              << "w8_acle_max_abs=" << w8_acle_error.max_abs << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
