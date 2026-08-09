#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"

#include "kai/kai_common.h"

#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using TritonKernel = void (*)(void *, void *, void *, void *, void *, int32_t,
                              int32_t, int32_t, int32_t, int32_t, int32_t);
using TritonWholeNKernel = void (*)(
    void *, void *, void *, void *, void *, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t);

struct SharedObject {
  explicit SharedObject(const char *path) {
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
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

void store_f16(uint8_t *destination, float value) {
  const uint16_t half = kai_cast_f16_f32(value);
  std::memcpy(destination, &half, sizeof(half));
}

uint16_t to_bf16_rne(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  // Preserve NaN as NaN rather than allowing the rounding carry to create an
  // infinity. The benchmark's generated values are finite, but keeping this
  // conversion complete makes the comparison contract explicit.
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

uint16_t ordered_bf16(uint16_t bits) {
  return static_cast<uint16_t>(
      bits & 0x8000u ? ~bits : static_cast<uint16_t>(bits | 0x8000u));
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 11 || argc == 7) {
    std::cerr << "usage: " << argv[0]
              << " M N K ITERATIONS [BATCHES [TRITON_SO BLOCK_N "
                 "[SYMBOL [kai_abi|kai_abi_bf16 [TRITON_BLOCK_M]]]]]\n";
    return 2;
  }

  try {
    const size_t m = std::stoull(argv[1]);
    const size_t n = std::stoull(argv[2]);
    const size_t k = std::stoull(argv[3]);
    const int iterations = std::stoi(argv[4]);
    const int batches = argc >= 6 ? std::stoi(argv[5]) : 11;
    const bool compare_triton = argc >= 8;
    const size_t block_n = compare_triton ? std::stoull(argv[7]) : 8;
    const std::string abi_mode = argc >= 10 ? argv[9] : "";
    const bool exact_kai_abi =
        abi_mode == "kai_abi" || abi_mode == "kai_abi_bf16";
    const bool triton_bf16_output = abi_mode == "kai_abi_bf16";
    constexpr size_t block_length = 32;
    constexpr size_t mr = 4;
    constexpr size_t nr = 4;
    constexpr size_t kr = 16;
    constexpr size_t sr = 2;
    const bool exact_tail_m = m < 16 && m % 4 == 0;
    const bool invalid_m =
        exact_kai_abi ? (m % 16 != 0 && !exact_tail_m) : (m % 16 != 0);
    if (m == 0 || n == 0 || k == 0 || invalid_m || n % nr != 0 ||
        k % block_length != 0 || n % block_n != 0 ||
        (block_n != 4 && block_n != 8) ||
        iterations <= 0 || batches <= 0) {
      throw std::runtime_error(
          "requires N%4=0, K%32=0 and M%16=0 "
          "(kai_abi also accepts M=4/8/12 tail objects)");
    }
    if (kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm() != mr ||
        kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm() != nr ||
        kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm() != kr ||
        kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm() != sr) {
      throw std::runtime_error("unexpected KleidiAI packing parameters");
    }

    std::vector<float> lhs(m * k);
    for (size_t index = 0; index < lhs.size(); ++index) {
      lhs[index] = std::sin(static_cast<float>(index) * 0.017f) * 1.7f +
                   std::cos(static_cast<float>(index) * 0.003f) * 0.31f;
    }

    const size_t groups = k / block_length;
    const size_t rhs_source_stride = groups * 18;
    std::vector<uint8_t> rhs_source(n * rhs_source_stride);
    for (size_t row = 0; row < n; ++row) {
      for (size_t group = 0; group < groups; ++group) {
        uint8_t *block =
            rhs_source.data() + row * rhs_source_stride + group * 18;
        store_f16(block,
                  0.001f + static_cast<float>((row * 7 + group * 3) % 29) *
                               0.00011f);
        for (size_t lane = 0; lane < 16; ++lane) {
          const uint8_t low = static_cast<uint8_t>(
              (row * 13 + group * 5 + lane * 3 + 1) % 16);
          const uint8_t high = static_cast<uint8_t>(
              (row * 11 + group * 7 + lane * 5 + 4) % 16);
          block[2 + lane] = static_cast<uint8_t>(low | (high << 4));
        }
      }
    }

    const size_t lhs_packed_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(
            m, k, block_length, mr, kr, sr);
    const size_t rhs_packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
            n, k, nr, kr, block_length);
    AlignedBuffer<uint8_t> lhs_packed(lhs_packed_size);
    AlignedBuffer<uint8_t> rhs_packed(rhs_packed_size);
    auto run_kleidiai_lhs_pack = [&] {
      kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(
          m, k, block_length, mr, kr, sr, 0, lhs.data(), k * sizeof(float),
          lhs_packed.data());
    };
    run_kleidiai_lhs_pack();
    const kai_rhs_pack_qs4cxs1s0_param rhs_pack_params{
        .lhs_zero_point = 1, .rhs_zero_point = 8};
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
        1, n, k, nr, kr, sr, block_length, rhs_source.data(), nullptr,
        rhs_packed.data(), 0, &rhs_pack_params);

    AlignedBuffer<float> output(m * n);
    const float clamp_min = -std::numeric_limits<float>::infinity();
    const float clamp_max = std::numeric_limits<float>::infinity();
    auto run_kleidiai = [&] {
      kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
          m, n, k, block_length, lhs_packed.data(), rhs_packed.data(),
          output.data(), n * sizeof(float), sizeof(float), clamp_min,
          clamp_max);
    };

    for (int iteration = 0; iteration < 100; ++iteration) {
      run_kleidiai();
    }
    std::vector<double> kai_samples;
    kai_samples.reserve(batches);

    double triton_median = 0.0;
    double triton_max_abs_error = 0.0;
    uint32_t triton_max_bf16_ulp = 0;
    size_t triton_bf16_mismatches = 0;
    if (compare_triton) {
      AlignedBuffer<int8_t> triton_lhs(m * k);
      AlignedBuffer<uint16_t> triton_lhs_scale(m * groups);
      AlignedBuffer<uint8_t> triton_rhs(n * k / 2);
      AlignedBuffer<uint16_t> triton_rhs_scale(groups * n);
      AlignedBuffer<float> triton_output(m * n);
      AlignedBuffer<uint16_t> triton_bf16(m * n);
      for (size_t index = 0; index < triton_lhs.count; ++index) {
        triton_lhs.data()[index] =
            static_cast<int8_t>((index * 29 + index / 97 + 7) % 255 - 127);
      }
      for (size_t index = 0; index < triton_lhs_scale.count; ++index) {
        triton_lhs_scale.data()[index] =
            kai_cast_f16_f32(0.001f + static_cast<float>(index % 17) * 0.0001f);
      }
      for (size_t index = 0; index < triton_rhs.count; ++index) {
        triton_rhs.data()[index] = static_cast<uint8_t>(
            ((index * 13 + 1) & 15) | (((index * 7 + 4) & 15) << 4));
      }
      for (size_t index = 0; index < triton_rhs_scale.count; ++index) {
        triton_rhs_scale.data()[index] =
            kai_cast_f16_f32(0.001f + static_cast<float>(index % 23) * 0.0001f);
      }

      SharedObject triton_library(argv[6]);
      const char *symbol =
          argc >= 9 ? argv[8] : "_w4_prefill_i8mm_kernel";
      const bool whole_n = std::string(symbol).find("whole_n") !=
                           std::string::npos;
      TritonKernel triton = nullptr;
      TritonWholeNKernel triton_whole_n = nullptr;
      if (whole_n)
        triton_whole_n =
            triton_library.symbol<TritonWholeNKernel>(symbol);
      else
        triton = triton_library.symbol<TritonKernel>(symbol);
      const size_t triton_block_m =
          argc >= 11 ? std::stoull(argv[10])
                     : (exact_kai_abi && m < 16 ? m : 16);
      if (m % triton_block_m != 0 ||
          (triton_block_m != 4 && triton_block_m != 8 &&
           triton_block_m != 12 && triton_block_m != 16))
        throw std::runtime_error("invalid TRITON_BLOCK_M");
      auto run_triton = [&] {
        for (int32_t pid_m = 0;
             pid_m < static_cast<int32_t>(m / triton_block_m);
             ++pid_m) {
          if (whole_n) {
            triton_whole_n(
                exact_kai_abi ? static_cast<void *>(lhs_packed.data())
                              : static_cast<void *>(triton_lhs.data()),
                exact_kai_abi
                    ? static_cast<void *>(lhs_packed.data())
                    : static_cast<void *>(triton_lhs_scale.data()),
                exact_kai_abi ? static_cast<void *>(rhs_packed.data())
                              : static_cast<void *>(triton_rhs.data()),
                exact_kai_abi
                    ? static_cast<void *>(rhs_packed.data())
                    : static_cast<void *>(triton_rhs_scale.data()),
                triton_bf16_output
                    ? static_cast<void *>(triton_bf16.data())
                    : static_cast<void *>(triton_output.data()),
                0, static_cast<int32_t>(n / block_n), pid_m, 0, 0,
                static_cast<int32_t>(m / triton_block_m), 1, 1);
            continue;
          }
          for (int32_t pid_n = 0;
               pid_n < static_cast<int32_t>(n / block_n); ++pid_n) {
            triton(exact_kai_abi
                       ? static_cast<void *>(lhs_packed.data())
                       : static_cast<void *>(triton_lhs.data()),
                   exact_kai_abi
                       ? static_cast<void *>(lhs_packed.data())
                       : static_cast<void *>(triton_lhs_scale.data()),
                   exact_kai_abi
                       ? static_cast<void *>(rhs_packed.data())
                       : static_cast<void *>(triton_rhs.data()),
                   exact_kai_abi
                       ? static_cast<void *>(rhs_packed.data())
                       : static_cast<void *>(triton_rhs_scale.data()),
                   triton_bf16_output
                       ? static_cast<void *>(triton_bf16.data())
                       : static_cast<void *>(triton_output.data()),
                   pid_m, pid_n, 0,
                   static_cast<int32_t>(m / 16),
                   static_cast<int32_t>(n / block_n), 1);
          }
        }
      };
      for (int iteration = 0; iteration < 100; ++iteration) {
        run_triton();
      }
      if (exact_kai_abi) {
        run_kleidiai();
        if (triton_bf16_output) {
          for (size_t index = 0; index < m * n; ++index) {
            const uint16_t expected = to_bf16_rne(output.data()[index]);
            const uint16_t actual = triton_bf16.data()[index];
            if (actual != expected)
              ++triton_bf16_mismatches;
            const uint32_t ulp = static_cast<uint32_t>(std::abs(
                static_cast<int>(ordered_bf16(actual)) -
                static_cast<int>(ordered_bf16(expected))));
            triton_max_bf16_ulp = std::max(triton_max_bf16_ulp, ulp);
          }
          if (triton_max_bf16_ulp > 4) {
            throw std::runtime_error("KAI ABI BF16 result exceeds 4 ULP: " +
                                     std::to_string(triton_max_bf16_ulp));
          }
        } else {
          for (size_t index = 0; index < m * n; ++index) {
            triton_max_abs_error = std::max(
                triton_max_abs_error,
                std::abs(static_cast<double>(output.data()[index]) -
                         static_cast<double>(triton_output.data()[index])));
          }
          if (triton_max_abs_error > 1.0e-5) {
            throw std::runtime_error("exact KAI ABI result mismatch: " +
                                     std::to_string(triton_max_abs_error));
          }
        }
      }
      std::vector<double> triton_samples;
      triton_samples.reserve(batches);
      for (int batch = 0; batch < batches; ++batch) {
        if ((batch & 1) == 0) {
          triton_samples.push_back(timed_us(run_triton, iterations));
          kai_samples.push_back(timed_us(run_kleidiai, iterations));
        } else {
          kai_samples.push_back(timed_us(run_kleidiai, iterations));
          triton_samples.push_back(timed_us(run_triton, iterations));
        }
      }
      std::sort(triton_samples.begin(), triton_samples.end());
      triton_median = triton_samples[triton_samples.size() / 2];
      // Keep the shared object and all Triton buffers alive until timing is
      // complete; their scope intentionally encloses the paired batches.
    } else {
      for (int batch = 0; batch < batches; ++batch) {
        kai_samples.push_back(timed_us(run_kleidiai, iterations));
      }
    }
    std::sort(kai_samples.begin(), kai_samples.end());
    const double median = kai_samples[kai_samples.size() / 2];
    std::vector<double> pack_samples;
    pack_samples.reserve(batches);
    for (int batch = 0; batch < batches; ++batch)
      pack_samples.push_back(timed_us(run_kleidiai_lhs_pack, iterations));
    std::sort(pack_samples.begin(), pack_samples.end());
    const double pack_median = pack_samples[pack_samples.size() / 2];
    const double equivalent_gops =
        2.0 * static_cast<double>(m) * static_cast<double>(n) *
        static_cast<double>(k) / median / 1.0e3;
    std::cout << std::setprecision(9)
              << "PASS KleidiAI Q4-prefill i8mm M=" << m << " N=" << n
              << " K=" << k << '\n'
              << "kernel_us=" << median << '\n'
              << "kleidiai_f32_lhs_pack_us=" << pack_median << '\n'
              << "equivalent_gops=" << equivalent_gops << '\n'
              << "pack_excluded=true\n";
    if (compare_triton) {
      std::cout << "triton_kernel_us=" << triton_median << '\n'
                << "triton_over_kleidiai=" << triton_median / median << "x\n"
                << "triton_exact_kai_abi="
                << (exact_kai_abi ? "true" : "false") << '\n';
      if (exact_kai_abi) {
        if (triton_bf16_output) {
          std::cout << "triton_output=bf16\n"
                    << "kleidiai_output=f32\n"
                    << "triton_bf16_mismatches="
                    << triton_bf16_mismatches << '\n'
                    << "triton_max_bf16_ulp=" << triton_max_bf16_ulp << '\n';
        } else {
          std::cout << "triton_max_abs_error=" << triton_max_abs_error << '\n';
        }
      }
      std::cout << "triton_grid_direct_calls=true\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
