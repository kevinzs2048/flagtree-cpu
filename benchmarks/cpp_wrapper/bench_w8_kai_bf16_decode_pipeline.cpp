#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

using PackKernel = void (*)(void *, void *, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t);
using MatrixKernel = void (*)(void *, void *, void *, void *, int32_t, int32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t);
using ProductionMatrixKernel = void (*)(void *, void *, void *, int32_t,
                                        int32_t, uint32_t, uint32_t, uint32_t,
                                        uint32_t, uint32_t, uint32_t);

struct SharedObject {
  explicit SharedObject(const std::string &path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(dlerror());
  }
  ~SharedObject() { dlclose(handle); }
  template <typename Function> Function symbol(const char *name) {
    dlerror();
    void *address = dlsym(handle, name);
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }
  template <typename Function> Function optional_symbol(const char *name) {
    dlerror();
    return reinterpret_cast<Function>(dlsym(handle, name));
  }
  void *handle;
};

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
              << " PACK_SO MATRIX_SO K N [ITERATIONS] [BATCHES]\n";
    return 2;
  }
  try {
    const size_t k = std::stoull(argv[3]);
    const size_t n = std::stoull(argv[4]);
    const int iterations = argc >= 6 ? std::stoi(argv[5]) : 100;
    const int batches = argc >= 7 ? std::stoi(argv[6]) : 21;
    if (k == 0 || n == 0 || k % 32 != 0 || n % 4 != 0 ||
        iterations <= 0 || batches <= 0)
      throw std::runtime_error("invalid K/N/repetition count");

    std::vector<uint16_t> input(k);
    std::vector<int8_t> weight(n * k);
    std::vector<float> weight_scale(n);
    std::vector<float> bias(n, 0.0f);
    for (size_t index = 0; index < k; ++index) {
      const float value = std::sin(static_cast<float>(index) * 0.017f) * 1.7f +
                          std::cos(static_cast<float>(index) * 0.003f) * 0.31f;
      input[index] = float_to_bf16(value);
    }
    for (size_t index = 0; index < weight.size(); ++index)
      weight[index] = static_cast<int8_t>((index * 29 + index / 97 + 7) % 255 -
                                          127);
    for (size_t column = 0; column < n; ++column)
      weight_scale[column] = 0.001f + (column % 31) * 0.00007f;

    const size_t mr =
        kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t nr =
        kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t sr =
        kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    if (mr != 1 || nr != 4 || kr != 8 || sr != 1)
      throw std::runtime_error("unexpected KAI microkernel geometry");

    const size_t lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(1, k, mr,
                                                                  kr, sr);
    const size_t rhs_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, nr,
                                                                  kr, sr);
    std::vector<uint8_t> triton_lhs(lhs_size);
    std::vector<uint8_t> kai_lhs(lhs_size);
    std::vector<uint8_t> rhs(rhs_size);
    std::vector<uint16_t> triton_output(n);
    std::vector<uint16_t> kai_output(n);
    std::vector<float> kai_f32(n);

    const kai_rhs_pack_qsi8cx_params rhs_params{.lhs_zero_point = 1,
                                                 .scale_multiplier = 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, n, k, nr, kr, sr, weight.data(), bias.data(), weight_scale.data(),
        rhs.data(), 0, &rhs_params);

    SharedObject pack_library(argv[1]);
    SharedObject matrix_library(argv[2]);
    PackKernel pack = pack_library.symbol<PackKernel>(
        "_pack_lhs_qai8dxp_bf16_kernel");
    MatrixKernel matrix = matrix_library.optional_symbol<MatrixKernel>(
        "_kai_w8_layout_pointer_bn8_kernel");
    const size_t matrix_nr = matrix ? 8 : 4;
    if (!matrix)
      matrix = matrix_library.optional_symbol<MatrixKernel>(
          "_kai_w8_layout_pointer_kernel");
    ProductionMatrixKernel production_matrix = nullptr;
    if (!matrix)
      production_matrix = matrix_library.symbol<ProductionMatrixKernel>(
          "_w8_qai8dxp_decode_sdot_kernel");
    const float clamp[2] = {-std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max()};

    auto run_triton = [&] {
      pack(input.data(), triton_lhs.data(), 0, 0, 0, 1, 1, 1);
      if (matrix) {
        matrix(triton_lhs.data(), rhs.data(), const_cast<float *>(clamp),
               triton_output.data(), 0, static_cast<int32_t>(n / matrix_nr), 0,
               0, 0, 1, 1, 1);
      } else {
        production_matrix(triton_lhs.data(), rhs.data(),
                          triton_output.data(), 0,
                          static_cast<int32_t>(n / 4), 0, 0, 0, 1, 1, 1);
      }
    };
    auto run_kai = [&] {
      kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
          1, k, mr, kr, sr, 0, input.data(), k * sizeof(uint16_t),
          kai_lhs.data());
      kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod(
          1, n, k, kai_lhs.data(), rhs.data(), kai_f32.data(),
          n * sizeof(float), sizeof(float), clamp[0], clamp[1]);
      for (size_t column = 0; column < n; ++column)
        kai_output[column] = float_to_bf16(kai_f32[column]);
    };

    run_triton();
    run_kai();
    if (std::memcmp(triton_lhs.data(), kai_lhs.data(), lhs_size) != 0)
      throw std::runtime_error("generated LHS pack is not bit-exact");
    if (std::memcmp(triton_output.data(), kai_output.data(),
                    n * sizeof(uint16_t)) != 0) {
      size_t mismatches = 0;
      for (size_t column = 0; column < n; ++column)
        mismatches += triton_output[column] != kai_output[column];
      throw std::runtime_error("BF16 output mismatch count=" +
                               std::to_string(mismatches));
    }

    const auto [triton_us, kai_us] = paired_medians_us(
        run_triton, run_kai, 20, iterations, batches);
    std::cout << std::setprecision(8)
              << "PASS exact-KAI W8 BF16 decode K=" << k << " N=" << n
              << '\n'
              << "triton_pipeline_us=" << triton_us << '\n'
              << "kleidiai_pipeline_us=" << kai_us << '\n'
              << "triton_over_kleidiai=" << triton_us / kai_us << "x\n"
              << "lhs_pack_bit_exact=true\n"
              << "bf16_output_bit_exact=true\n"
              << "matrix_nr=" << matrix_nr << '\n'
              << "production_abi=" << (production_matrix != nullptr) << '\n'
              << "python_excluded=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
