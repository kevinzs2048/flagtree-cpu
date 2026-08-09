#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using TritonKernel = void (*)(void *, void *, void *, void *, int32_t, int32_t,
                              int32_t, int32_t, int32_t, int32_t);

struct SharedObject {
  explicit SharedObject(const char *path) {
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
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

template <typename T> struct AlignedBuffer {
  explicit AlignedBuffer(size_t count) : count(count) {
    void *storage = nullptr;
    if (posix_memalign(&storage, 64, count * sizeof(T)) != 0)
      throw std::bad_alloc();
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
double timedUs(Function &&function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration)
    function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 6 || argc > 7) {
    std::cerr << "usage: " << argv[0]
              << " TRITON_SO M N K ITERATIONS [BATCHES]\n";
    return 2;
  }
  try {
    const size_t m = std::stoull(argv[2]);
    const size_t n = std::stoull(argv[3]);
    const size_t k = std::stoull(argv[4]);
    const int iterations = std::stoi(argv[5]);
    const int batches = argc == 7 ? std::stoi(argv[6]) : 11;
    constexpr size_t mr = 4;
    constexpr size_t nr = 4;
    constexpr size_t kr = 8;
    constexpr size_t sr = 1;
    if (m == 0 || n == 0 || k == 0 || m % 16 != 0 || n % 4 != 0 ||
        k % 32 != 0 || iterations <= 0 || batches <= 0)
      throw std::runtime_error("requires M%16=0, N%4=0 and K%32=0");
    if (kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm() !=
            mr ||
        kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm() !=
            nr ||
        kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm() !=
            kr ||
        kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm() !=
            sr)
      throw std::runtime_error("unexpected KleidiAI packing parameters");

    std::vector<float> lhs(m * k);
    std::vector<int8_t> rhs(n * k);
    std::vector<float> rhsScale(n);
    std::vector<float> bias(n);
    for (size_t index = 0; index < lhs.size(); ++index)
      lhs[index] = std::sin(static_cast<float>(index) * 0.017f) * 1.7f +
                   std::cos(static_cast<float>(index) * 0.003f) * 0.31f;
    for (size_t index = 0; index < rhs.size(); ++index)
      rhs[index] =
          static_cast<int8_t>((index * 29 + index / 97 + 7) % 255 - 127);
    for (size_t column = 0; column < n; ++column) {
      rhsScale[column] = 0.001f + (column % 31) * 0.00007f;
      bias[column] = (static_cast<int>(column % 17) - 8) * 0.015f;
    }

    const size_t lhsSize =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
    const size_t rhsSize =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, nr, kr,
                                                                 sr);
    AlignedBuffer<uint8_t> lhsPacked(lhsSize);
    AlignedBuffer<uint8_t> rhsPacked(rhsSize);
    kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr, 0, lhs.data(),
                                       k * sizeof(float), lhsPacked.data());
    const kai_rhs_pack_qsi8cx_params params{.lhs_zero_point = 1,
                                            .scale_multiplier = 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1, n, k, nr, kr, sr, rhs.data(),
                                             bias.data(), rhsScale.data(),
                                             rhsPacked.data(), 0, &params);

    AlignedBuffer<float> tritonOutput(m * n);
    AlignedBuffer<float> kaiOutput(m * n);
    const float clamp[2] = {-7.0f, 6.0f};
    SharedObject library(argv[1]);
    TritonKernel triton =
        library.symbol<TritonKernel>("_kai_w8_prefill_kernel");
    auto runTriton = [&] {
      for (int32_t pidM = 0; pidM < static_cast<int32_t>(m / 16); ++pidM)
        for (int32_t pidN = 0; pidN < static_cast<int32_t>(n / 4); ++pidN)
          triton(lhsPacked.data(), rhsPacked.data(), const_cast<float *>(clamp),
                 tritonOutput.data(), pidM, pidN, 0,
                 static_cast<int32_t>(m / 16), static_cast<int32_t>(n / 4), 1);
    };
    auto runKai = [&] {
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
          m, n, k, lhsPacked.data(), rhsPacked.data(), kaiOutput.data(),
          n * sizeof(float), sizeof(float), clamp[0], clamp[1]);
    };
    runTriton();
    runKai();
    double maxAbs = 0.0;
    size_t mismatches = 0;
    for (size_t index = 0; index < m * n; ++index) {
      const double error =
          std::abs(static_cast<double>(tritonOutput.data()[index]) -
                   static_cast<double>(kaiOutput.data()[index]));
      maxAbs = std::max(maxAbs, error);
      mismatches += error > 2.0e-5;
    }
    if (mismatches)
      throw std::runtime_error(
          "output mismatch count=" + std::to_string(mismatches) +
          " max_abs=" + std::to_string(maxAbs));

    for (int iteration = 0; iteration < 100; ++iteration) {
      runTriton();
      runKai();
    }
    std::vector<double> tritonSamples;
    std::vector<double> kaiSamples;
    for (int batch = 0; batch < batches; ++batch) {
      if ((batch & 1) == 0) {
        tritonSamples.push_back(timedUs(runTriton, iterations));
        kaiSamples.push_back(timedUs(runKai, iterations));
      } else {
        kaiSamples.push_back(timedUs(runKai, iterations));
        tritonSamples.push_back(timedUs(runTriton, iterations));
      }
    }
    std::sort(tritonSamples.begin(), tritonSamples.end());
    std::sort(kaiSamples.begin(), kaiSamples.end());
    const double tritonUs = tritonSamples[tritonSamples.size() / 2];
    const double kaiUs = kaiSamples[kaiSamples.size() / 2];
    std::cout << std::setprecision(9)
              << "PASS exact-KAI-layout W8-prefill M=" << m << " N=" << n
              << " K=" << k << '\n'
              << "triton_kernel_us=" << tritonUs << '\n'
              << "kleidiai_kernel_us=" << kaiUs << '\n'
              << "triton_over_kleidiai=" << tritonUs / kaiUs << "x\n"
              << "max_abs_error=" << maxAbs << '\n'
              << "pack_excluded=true\n"
              << "triton_grid_direct_calls=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
