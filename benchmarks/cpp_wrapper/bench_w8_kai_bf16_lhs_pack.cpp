#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using TritonPack = void (*)(void *, void *, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t);

uint16_t float_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

template <typename Function>
double timed_us(Function &&function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration)
    function();
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

} // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 7) {
    std::cerr << "usage: " << argv[0]
              << " TRITON_SO M K MR [ITERATIONS] [BATCHES]\n";
    return 2;
  }
  try {
    const size_t m = std::stoull(argv[2]);
    const size_t k = std::stoull(argv[3]);
    const size_t mr = std::stoull(argv[4]);
    const int iterations = argc >= 6 ? std::stoi(argv[5]) : 1000;
    const int batches = argc >= 7 ? std::stoi(argv[6]) : 21;
    if (m == 0 || k == 0 || k % 32 != 0 || (mr != 1 && mr != 4) ||
        m % mr != 0 || iterations <= 0 || batches <= 0) {
      throw std::runtime_error("invalid M/K/MR/repetition count");
    }

    const bool finite_sweep = std::getenv("KAI_BF16_FINITE_SWEEP") != nullptr;
    std::vector<uint16_t> input(m * k);
    if (finite_sweep) {
      size_t index = 0;
      for (uint32_t bits = 0; bits <= 0xffffu && index < input.size(); ++bits) {
        if ((bits & 0x7f80u) != 0x7f80u)
          input[index++] = static_cast<uint16_t>(bits);
      }
      if (index < 65280)
        throw std::runtime_error(
            "finite sweep requires space for all 65,280 BF16 values");
      std::fill(input.begin() + static_cast<std::ptrdiff_t>(index),
                input.end(), uint16_t{0});
    } else {
      for (size_t index = 0; index < input.size(); ++index) {
        const float value =
            std::sin(static_cast<float>(index) * 0.017f) * 1.7f +
            std::cos(static_cast<float>(index) * 0.003f) * 0.31f +
            static_cast<float>(index % 11) * 0.001f;
        input[index] = float_to_bf16(value);
      }
    }

    const size_t packed_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(
            m, k, mr, 8, 1);
    std::vector<uint8_t> triton_output(packed_size);
    std::vector<uint8_t> kai_output(packed_size);

    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
      throw std::runtime_error(dlerror());
    dlerror();
    auto triton = reinterpret_cast<TritonPack>(
        dlsym(library, "_pack_lhs_qai8dxp_bf16_mr4_kernel"));
    uint32_t rows_per_program = triton ? 4 : 1;
    dlerror();
    if (!triton) {
      triton = reinterpret_cast<TritonPack>(
          dlsym(library, "_pack_lhs_qai8dxp_bf16_mr2_kernel"));
      if (triton)
        rows_per_program = 2;
      dlerror();
    }
    if (!triton) {
      triton = reinterpret_cast<TritonPack>(
          dlsym(library, "_pack_lhs_qai8dxp_bf16_kernel"));
      if (const char *error = dlerror())
        throw std::runtime_error(error);
    }

    auto run_triton = [&] {
      const uint32_t programs =
          static_cast<uint32_t>(m / rows_per_program);
      for (uint32_t program = 0; program < programs; ++program)
        triton(input.data(), triton_output.data(), program, 0, 0, programs, 1,
               1);
    };
    auto run_kai = [&] {
      kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
          m, k, mr, 8, 1, 0, input.data(), k * sizeof(uint16_t),
          kai_output.data());
    };

    run_triton();
    run_kai();
    if (std::memcmp(triton_output.data(), kai_output.data(), packed_size) != 0) {
      size_t mismatches = 0;
      size_t first = packed_size;
      for (size_t index = 0; index < packed_size; ++index) {
        if (triton_output[index] != kai_output[index]) {
          ++mismatches;
          first = std::min(first, index);
        }
      }
      throw std::runtime_error("packed output mismatch count=" +
                               std::to_string(mismatches) + " first=" +
                               std::to_string(first));
    }

    const auto [triton_us, kai_us] = paired_medians_us(
        run_triton, run_kai, 50, iterations, batches);
    std::cout << std::setprecision(8)
              << "PASS KAI BF16 LHS pack M=" << m << " K=" << k
              << " MR=" << mr << '\n'
              << "triton_pack_us=" << triton_us << '\n'
              << "kleidiai_pack_us=" << kai_us << '\n'
              << "triton_over_kleidiai=" << triton_us / kai_us << "x\n"
              << "packed_bit_exact=true\n"
              << "input_mode=" << (finite_sweep ? "all-finite-bf16" : "wave")
              << '\n'
              << "schedule=mr" << rows_per_program << '\n'
              << "packed_bytes=" << packed_size << '\n';
    dlclose(library);
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
