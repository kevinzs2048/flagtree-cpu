#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using TritonKernel = void (*)(void *, void *, void *, void *, int32_t,
                              int32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t);

struct SharedObject {
  explicit SharedObject(const std::string &path) {
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

  template <typename Function> Function symbol(const char *name) {
    dlerror();
    void *address = dlsym(handle, name);
    if (const char *error = dlerror()) {
      throw std::runtime_error(error);
    }
    return reinterpret_cast<Function>(address);
  }

  void *handle = nullptr;
};

template <typename T> struct AlignedBuffer {
  explicit AlignedBuffer(size_t count) : count(count) {
    void *storage = nullptr;
    if (posix_memalign(&storage, 64, count * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    pointer = static_cast<T *>(storage);
  }

  ~AlignedBuffer() { std::free(pointer); }
  AlignedBuffer(const AlignedBuffer &) = delete;
  AlignedBuffer &operator=(const AlignedBuffer &) = delete;
  T *data() { return pointer; }
  const T *data() const { return pointer; }
  T &operator[](size_t index) { return pointer[index]; }
  const T &operator[](size_t index) const { return pointer[index]; }

  size_t count;
  T *pointer = nullptr;
};

template <typename Function>
double timed_us(Function &&function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    function();
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}

template <typename First, typename Second>
std::pair<double, double> paired_medians_us(First &&first, Second &&second,
                                            int warmup, int iterations,
                                            int batches) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    first();
    second();
  }

  std::vector<double> first_samples;
  std::vector<double> second_samples;
  first_samples.reserve(batches);
  second_samples.reserve(batches);
  for (int batch = 0; batch < batches; ++batch) {
    if ((batch & 1) == 0) {
      first_samples.push_back(timed_us(first, iterations));
      second_samples.push_back(timed_us(second, iterations));
    } else {
      second_samples.push_back(timed_us(second, iterations));
      first_samples.push_back(timed_us(first, iterations));
    }
  }
  std::sort(first_samples.begin(), first_samples.end());
  std::sort(second_samples.begin(), second_samples.end());
  return {first_samples[first_samples.size() / 2],
          second_samples[second_samples.size() / 2]};
}

void store_f16(uint8_t *destination, float value) {
  const uint16_t half = kai_cast_f16_f32(value);
  std::memcpy(destination, &half, sizeof(half));
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 6) {
    std::cerr << "usage: " << argv[0]
              << " TRITON_SO K N ITERATIONS [BATCHES]\n";
    return 2;
  }

  try {
    const size_t k = std::stoull(argv[2]);
    const size_t n = std::stoull(argv[3]);
    const int iterations = std::stoi(argv[4]);
    const int batches = argc >= 6 ? std::stoi(argv[5]) : 9;
    constexpr size_t block_length = 32;
    if (k == 0 || n == 0 || k % block_length != 0 || n % 4 != 0 ||
        iterations <= 0 || batches <= 0) {
      throw std::runtime_error("invalid K/N/repetition count");
    }

    const size_t mr =
        kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t nr =
        kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t sr =
        kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    if (mr != 1 || nr != 4 || kr != 16 || sr != 2) {
      throw std::runtime_error("unexpected KleidiAI packing parameters");
    }

    std::vector<float> lhs(k);
    for (size_t index = 0; index < k; ++index) {
      lhs[index] = std::sin(static_cast<float>(index) * 0.017f) * 1.7f +
                   std::cos(static_cast<float>(index) * 0.003f) * 0.31f;
    }

    // KAI's unpacked RHS format stores, for every N row and K32 block,
    // fp16(scale) followed by 16 bytes of offset-binary uint4 values.  Let the
    // official packer produce the final microkernel layout used by both paths.
    const size_t groups = k / block_length;
    const size_t rhs_source_stride = groups * 18;
    std::vector<uint8_t> rhs_source(n * rhs_source_stride);
    for (size_t row = 0; row < n; ++row) {
      for (size_t group = 0; group < groups; ++group) {
        uint8_t *block = rhs_source.data() + row * rhs_source_stride + group * 18;
        store_f16(block, 0.001f + static_cast<float>((row * 7 + group * 3) % 29) *
                                     0.00011f);
        for (size_t lane = 0; lane < 16; ++lane) {
          const int low_signed =
              static_cast<int>((row * 13 + group * 5 + lane * 3 + 1) % 16) - 8;
          const int high_signed =
              static_cast<int>((row * 11 + group * 7 + lane * 5 + 4) % 16) - 8;
          const uint8_t low = static_cast<uint8_t>(low_signed + 8);
          const uint8_t high = static_cast<uint8_t>(high_signed + 8);
          block[2 + lane] = static_cast<uint8_t>(low | (high << 4));
        }
      }
    }

    const size_t lhs_packed_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(
            1, k, block_length, mr, kr, sr);
    const size_t rhs_packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
            n, k, nr, kr, block_length);
    AlignedBuffer<uint8_t> lhs_packed(lhs_packed_size);
    AlignedBuffer<uint8_t> rhs_packed(rhs_packed_size);
    kai_run_lhs_quant_pack_qsi8d32p_f32(
        1, k, block_length, mr, kr, sr, 0, lhs.data(), k * sizeof(float),
        lhs_packed.data());
    const kai_rhs_pack_qs4cxs1s0_param rhs_pack_params{
        .lhs_zero_point = 1, .rhs_zero_point = 8};
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
        1, n, k, nr, kr, sr, block_length, rhs_source.data(), nullptr,
        rhs_packed.data(), 0, &rhs_pack_params);

    const float clamp[2] = {-7.0f, 6.0f};
    AlignedBuffer<float> triton_output(n);
    AlignedBuffer<float> kleidiai_output(n);
    SharedObject triton_library(argv[1]);
    TritonKernel triton = triton_library.symbol<TritonKernel>(
        "_kai_w4_layout_split_kernel");

    auto run_triton = [&] {
      triton(lhs_packed.data(), rhs_packed.data(), const_cast<float *>(clamp),
             triton_output.data(), 0, static_cast<int32_t>(n / 4), 0, 0, 0,
             1, 1, 1);
    };
    auto run_kleidiai = [&] {
      kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
          1, n, k, block_length, lhs_packed.data(), rhs_packed.data(),
          kleidiai_output.data(), n * sizeof(float), sizeof(float), clamp[0],
          clamp[1]);
    };

    run_triton();
    run_kleidiai();
    size_t mismatches = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    for (size_t column = 0; column < n; ++column) {
      const double actual = triton_output[column];
      const double expected = kleidiai_output[column];
      const double abs_error = std::abs(actual - expected);
      const double rel_error =
          abs_error / std::max(std::abs(expected), 1.0e-12);
      max_abs_error = std::max(max_abs_error, abs_error);
      max_rel_error = std::max(max_rel_error, rel_error);
      mismatches += abs_error > 3.0e-5 + 3.0e-6 * std::abs(expected);
    }
    if (mismatches != 0) {
      throw std::runtime_error("Triton/KleidiAI output mismatch count=" +
                               std::to_string(mismatches) +
                               " max_abs=" + std::to_string(max_abs_error) +
                               " max_rel=" + std::to_string(max_rel_error));
    }

    const auto [triton_us, kleidiai_us] = paired_medians_us(
        run_triton, run_kleidiai, 100, iterations, batches);
    std::cout << std::setprecision(8)
              << "PASS exact-KAI-layout W4 K=" << k << " N=" << n << '\n'
              << "triton_kernel_us=" << triton_us << '\n'
              << "kleidiai_kernel_us=" << kleidiai_us << '\n'
              << "triton_over_kleidiai=" << triton_us / kleidiai_us << "x\n"
              << "max_abs_error=" << max_abs_error << '\n'
              << "max_rel_error=" << max_rel_error << '\n'
              << "pack_excluded=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
