#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
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
                              int32_t, int32_t, int32_t, int32_t);
using ProductionMatrixKernel = void (*)(void *, void *, void *, uint32_t,
                                        uint32_t, uint32_t, uint32_t,
                                        uint32_t, uint32_t);

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

bool active_lhs_rows_equal(const std::vector<uint8_t> &generated,
                           const std::vector<uint8_t> &reference, size_t m,
                           size_t k) {
  const size_t panel_stride = 4 * (k + 8);
  for (size_t row = 0; row < m; ++row) {
    const size_t panel = row / 4;
    const size_t panel_row = row % 4;
    for (size_t offset = 0; offset < k; offset += 8) {
      const size_t packed_offset = panel * panel_stride + (offset / 8) * 32 +
                                   panel_row * 8;
      if (std::memcmp(generated.data() + packed_offset,
                      reference.data() + packed_offset, 8) != 0)
        return false;
    }
    const size_t metadata = panel * panel_stride + 4 * k;
    if (std::memcmp(generated.data() + metadata + panel_row * 4,
                    reference.data() + metadata + panel_row * 4, 4) != 0 ||
        std::memcmp(generated.data() + metadata + 16 + panel_row * 4,
                    reference.data() + metadata + 16 + panel_row * 4, 4) != 0)
      return false;
  }
  return true;
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
  if (argc < 6 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " PACK_SO MATRIX_SO M N K [ITERATIONS] [BATCHES]\n";
    return 2;
  }
  try {
    const size_t m = std::stoull(argv[3]);
    const size_t n = std::stoull(argv[4]);
    const size_t k = std::stoull(argv[5]);
    const int iterations = argc >= 7 ? std::stoi(argv[6]) : 100;
    const int batches = argc >= 8 ? std::stoi(argv[7]) : 21;
    constexpr size_t mr = 4;
    constexpr size_t nr = 4;
    constexpr size_t kr = 8;
    constexpr size_t sr = 1;
    if (m < 2 || m > 16 || n == 0 || k == 0 || n % 4 != 0 ||
        k % 32 != 0 || iterations <= 0 || batches <= 0)
      throw std::runtime_error("requires 2<=M<=16, N%4=0 and K%32=0");

    std::vector<uint16_t> input(m * k);
    std::vector<int8_t> weight(n * k);
    std::vector<float> weight_scale(n);
    std::vector<float> bias(n, 0.0f);
    for (size_t index = 0; index < input.size(); ++index) {
      const float value = std::sin(static_cast<float>(index) * 0.017f) * 1.7f +
                          std::cos(static_cast<float>(index) * 0.003f) * 0.31f;
      input[index] = float_to_bf16(value);
    }
    for (size_t index = 0; index < weight.size(); ++index)
      weight[index] = static_cast<int8_t>((index * 29 + index / 97 + 7) % 255 -
                                          127);
    for (size_t column = 0; column < n; ++column)
      weight_scale[column] = 0.001f + (column % 31) * 0.00007f;

    const size_t lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(m, k, mr,
                                                                  kr, sr);
    const size_t rhs_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, nr,
                                                                  kr, sr);
    std::vector<uint8_t> triton_lhs(lhs_size);
    std::vector<uint8_t> kai_lhs(lhs_size);
    std::vector<uint8_t> rhs(rhs_size);
    const size_t triton_rows = m <= 4 ? 4 : m <= 8 ? 8 : m <= 12 ? 12 : 16;
    std::vector<uint16_t> triton_output(triton_rows * n);
    std::vector<uint16_t> kai_output(m * n);
    std::vector<float> kai_f32(m * n);

    const kai_rhs_pack_qsi8cx_params rhs_params{.lhs_zero_point = 1,
                                                 .scale_multiplier = 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, n, k, nr, kr, sr, weight.data(), bias.data(), weight_scale.data(),
        rhs.data(), 0, &rhs_params);

    SharedObject pack_library(argv[1]);
    SharedObject matrix_library(argv[2]);
    PackKernel pack = pack_library.optional_symbol<PackKernel>(
        "_pack_lhs_qai8dxp_bf16_mr4_kernel");
    uint32_t rows_per_pack_program = pack ? 4 : 1;
    if (!pack) {
      pack = pack_library.optional_symbol<PackKernel>(
          "_pack_lhs_qai8dxp_bf16_mr2_kernel");
      if (pack)
        rows_per_pack_program = 2;
    }
    if (!pack)
      pack = pack_library.symbol<PackKernel>(
          "_pack_lhs_qai8dxp_bf16_kernel");
    const char *matrix_symbol =
        m <= 8    ? "_kai_w8_prefill_short_tail_kernel"
        : m == 12 ? "_kai_w8_prefill_m12_tail_kernel"
                  : "_kai_w8_prefill_kernel";
    MatrixKernel matrix = matrix_library.optional_symbol<MatrixKernel>(
        matrix_symbol);
    ProductionMatrixKernel production_matrix = nullptr;
    if (!matrix) {
      const char *production_symbol =
          m <= 8    ? "_w8_qai8dxp_prefill_short_tail_kernel"
          : m <= 12 ? "_w8_qai8dxp_prefill_m12_kernel"
                    : "_w8_qai8dxp_prefill_i8mm_kernel";
      production_matrix = matrix_library.symbol<ProductionMatrixKernel>(
          production_symbol);
    }
    const float clamp[2] = {-std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max()};

    auto run_triton = [&] {
      const uint32_t pack_programs =
          static_cast<uint32_t>((m + rows_per_pack_program - 1) /
                                rows_per_pack_program);
      for (uint32_t program = 0; program < pack_programs; ++program)
        pack(input.data(), triton_lhs.data(), program, 0, 0, pack_programs, 1,
             1);
      for (int32_t pid_n = 0; pid_n < static_cast<int32_t>(n / 4); ++pid_n) {
        if (matrix) {
          matrix(triton_lhs.data(), rhs.data(), const_cast<float *>(clamp),
                 triton_output.data(), 0, pid_n, 0, 1,
                 static_cast<int32_t>(n / 4), 1);
        } else {
          production_matrix(triton_lhs.data(), rhs.data(),
                            triton_output.data(), 0, pid_n, 0, 1,
                            static_cast<uint32_t>(n / 4), 1);
        }
      }
    };
    auto run_kai = [&] {
      kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
          m, k, mr, kr, sr, 0, input.data(), k * sizeof(uint16_t),
          kai_lhs.data());
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
          m, n, k, kai_lhs.data(), rhs.data(), kai_f32.data(),
          n * sizeof(float), sizeof(float), clamp[0], clamp[1]);
      for (size_t index = 0; index < m * n; ++index)
        kai_output[index] = float_to_bf16(kai_f32[index]);
    };
    auto run_triton_pack = [&] {
      const uint32_t pack_programs =
          static_cast<uint32_t>((m + rows_per_pack_program - 1) /
                                rows_per_pack_program);
      for (uint32_t program = 0; program < pack_programs; ++program)
        pack(input.data(), triton_lhs.data(), program, 0, 0, pack_programs, 1,
             1);
    };
    auto run_kai_pack = [&] {
      kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
          m, k, mr, kr, sr, 0, input.data(), k * sizeof(uint16_t),
          kai_lhs.data());
    };
    auto run_triton_matrix = [&] {
      for (int32_t pid_n = 0; pid_n < static_cast<int32_t>(n / 4); ++pid_n) {
        if (matrix) {
          matrix(triton_lhs.data(), rhs.data(), const_cast<float *>(clamp),
                 triton_output.data(), 0, pid_n, 0, 1,
                 static_cast<int32_t>(n / 4), 1);
        } else {
          production_matrix(triton_lhs.data(), rhs.data(),
                            triton_output.data(), 0, pid_n, 0, 1,
                            static_cast<uint32_t>(n / 4), 1);
        }
      }
    };
    auto run_kai_matrix = [&] {
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
          m, n, k, kai_lhs.data(), rhs.data(), kai_f32.data(),
          n * sizeof(float), sizeof(float), clamp[0], clamp[1]);
      for (size_t index = 0; index < m * n; ++index)
        kai_output[index] = float_to_bf16(kai_f32[index]);
    };

    run_triton();
    run_kai();
    if (!active_lhs_rows_equal(triton_lhs, kai_lhs, m, k))
      throw std::runtime_error("generated LHS pack is not bit-exact");
    if (std::memcmp(triton_output.data(), kai_output.data(),
                    m * n * sizeof(uint16_t)) != 0) {
      size_t mismatches = 0;
      for (size_t index = 0; index < m * n; ++index)
        mismatches += triton_output[index] != kai_output[index];
      throw std::runtime_error("BF16 output mismatch count=" +
                               std::to_string(mismatches));
    }

    const auto [triton_us, kai_us] = paired_medians_us(
        run_triton, run_kai, 20, iterations, batches);
    const auto [triton_pack_us, kai_pack_us] = paired_medians_us(
        run_triton_pack, run_kai_pack, 20, iterations, batches);
    const auto [triton_matrix_us, kai_matrix_us] = paired_medians_us(
        run_triton_matrix, run_kai_matrix, 20, iterations, batches);
    std::cout << std::setprecision(8)
              << "PASS exact-KAI W8 BF16 prefill M=" << m << " N=" << n
              << " K=" << k << '\n'
              << "triton_pipeline_us=" << triton_us << '\n'
              << "kleidiai_pipeline_us=" << kai_us << '\n'
              << "triton_over_kleidiai=" << triton_us / kai_us << "x\n"
              << "triton_pack_us=" << triton_pack_us << '\n'
              << "kleidiai_pack_us=" << kai_pack_us << '\n'
              << "triton_matrix_bf16_us=" << triton_matrix_us << '\n'
              << "kleidiai_matrix_bf16_us=" << kai_matrix_us << '\n'
              << "lhs_pack_bit_exact=true\n"
              << "bf16_output_bit_exact=true\n"
              << "pack_schedule=mr" << rows_per_pack_program << '\n'
              << "production_abi=" << (production_matrix != nullptr) << '\n'
              << "python_excluded=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
