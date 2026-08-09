#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

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

std::string dirname(const std::string &path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 10) {
    std::cerr << "usage: " << argv[0]
              << " TRITON_SO K N ITERATIONS [BATCHES] [BLOCK_N] "
                 "[auto|dot|reduce|partial|split|pointer|outer-pointer|"
                 "pointer8] "
                 "[REFERENCE_TRITON_SO] [REFERENCE_MODE]\n";
    return 2;
  }

  try {
    const size_t k = std::stoull(argv[2]);
    const size_t n = std::stoull(argv[3]);
    const int iterations = std::stoi(argv[4]);
    const int batches = argc >= 6 ? std::stoi(argv[5]) : 9;
    const size_t block_n = argc >= 7 ? std::stoull(argv[6]) : 4;
    const std::string mode = argc >= 8 ? argv[7] : "auto";
    if (k == 0 || n == 0 || k % 32 != 0 || n % block_n != 0 ||
        block_n % 4 != 0 || iterations <= 0 || batches <= 0 ||
        (mode != "auto" && mode != "dot" && mode != "reduce" &&
         mode != "partial" && mode != "split" && mode != "pointer" &&
         mode != "outer-pointer" && mode != "pointer8") ||
        (mode == "pointer8" && block_n != 8) ||
        (mode != "dot" && mode != "auto" && mode != "pointer8" &&
         block_n != 4) ||
        (mode == "auto" && block_n != 4)) {
      throw std::runtime_error("invalid K/N/BLOCK_N/repetition count");
    }

    const size_t mr =
        kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t nr =
        kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t sr =
        kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    if (mr != 1 || nr != 4 || kr != 8 || sr != 1) {
      throw std::runtime_error("unexpected KleidiAI packing parameters");
    }

    std::vector<float> lhs(k);
    std::vector<int8_t> rhs(n * k);
    std::vector<float> rhs_scale(n);
    std::vector<float> bias(n);
    for (size_t index = 0; index < k; ++index) {
      lhs[index] = std::sin(static_cast<float>(index) * 0.017f) * 1.7f +
                   std::cos(static_cast<float>(index) * 0.003f) * 0.31f;
    }
    for (size_t index = 0; index < rhs.size(); ++index) {
      rhs[index] = static_cast<int8_t>((index * 29 + index / 97 + 7) % 255 -
                                       127);
    }
    for (size_t column = 0; column < n; ++column) {
      rhs_scale[column] = 0.001f + (column % 31) * 0.00007f;
      bias[column] = (static_cast<int>(column % 17) - 8) * 0.015f;
    }

    const size_t lhs_packed_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(1, k, mr, kr, sr);
    const size_t rhs_packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, nr,
                                                                  kr, sr);
    AlignedBuffer<uint8_t> lhs_packed(lhs_packed_size);
    AlignedBuffer<uint8_t> rhs_packed(rhs_packed_size);
    kai_run_lhs_quant_pack_qai8dxp_f32(1, k, mr, kr, sr, 0, lhs.data(),
                                       k * sizeof(float), lhs_packed.data());
    const kai_rhs_pack_qsi8cx_params rhs_pack_params{
        .lhs_zero_point = 1, .scale_multiplier = 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, n, k, nr, kr, sr, rhs.data(), bias.data(), rhs_scale.data(),
        rhs_packed.data(), 0, &rhs_pack_params);

    const float clamp[2] = {-7.0f, 6.0f};
    AlignedBuffer<float> triton_output(n);
    AlignedBuffer<float> reference_output(n);
    AlignedBuffer<float> kleidiai_output(n);
    SharedObject triton_library(argv[1]);
    const std::string selected_mode = mode == "auto" ? "pointer" : mode;
    auto symbol_for_mode = [](const std::string &kernel_mode) {
      return kernel_mode == "reduce"    ? "_kai_w8_layout_reduce_kernel"
             : kernel_mode == "partial" ? "_kai_w8_layout_partial_kernel"
             : kernel_mode == "pointer8"
                 ? "_kai_w8_layout_pointer_bn8_kernel"
             : kernel_mode == "outer-pointer"
                 ? "_kai_w8_layout_outer_pointer_kernel"
             : kernel_mode == "pointer" ? "_kai_w8_layout_pointer_kernel"
             : kernel_mode == "split"   ? "_kai_w8_layout_split_kernel"
                                         : "_kai_w8_layout_gemv_kernel";
    };
    const char *symbol = symbol_for_mode(selected_mode);
    TritonKernel triton = triton_library.symbol<TritonKernel>(symbol);
    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> jit_wrapper(
        dirname(argv[1]), symbol);
    std::unique_ptr<SharedObject> reference_library;
    TritonKernel reference = nullptr;
    if (argc >= 9) {
      reference_library = std::make_unique<SharedObject>(argv[8]);
      const std::string reference_mode = argc == 10 ? argv[9] : selected_mode;
      if (reference_mode != "dot" && reference_mode != "reduce" &&
          reference_mode != "partial" && reference_mode != "split" &&
          reference_mode != "pointer" &&
          reference_mode != "outer-pointer" &&
          reference_mode != "pointer8") {
        throw std::runtime_error("invalid reference mode");
      }
      reference = reference_library->symbol<TritonKernel>(
          symbol_for_mode(reference_mode));
    }

    auto run_triton = [&] {
      triton(lhs_packed.data(), rhs_packed.data(), const_cast<float *>(clamp),
             triton_output.data(), 0, static_cast<int32_t>(n / block_n), 0, 0,
             0, 1, 1, 1);
    };
    auto run_kleidiai = [&] {
      kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod(
          1, n, k, lhs_packed.data(), rhs_packed.data(),
          kleidiai_output.data(), n * sizeof(float), sizeof(float), clamp[0],
          clamp[1]);
    };
    void *lhs_pointer = lhs_packed.data();
    void *rhs_pointer = rhs_packed.data();
    void *clamp_pointer = const_cast<float *>(clamp);
    void *output_pointer = triton_output.data();
    int32_t first_block = 0;
    int32_t last_block = static_cast<int32_t>(n / block_n);
    void *wrapper_args[] = {&lhs_pointer, &rhs_pointer, &clamp_pointer,
                            &output_pointer, &first_block, &last_block};
    auto run_wrapper = [&] {
      jit_wrapper.launch_with_signature(
          1, 1, 1, 1, nullptr, wrapper_args,
          "*i8,*i8,*fp32,*fp32,i32,i32", 6);
    };
    auto run_reference = [&] {
      reference(lhs_packed.data(), rhs_packed.data(),
                const_cast<float *>(clamp), reference_output.data(), 0,
                static_cast<int32_t>(n / block_n), 0, 0, 0, 1, 1, 1);
    };

    run_triton();
    run_wrapper();
    if (reference)
      run_reference();
    run_kleidiai();
    if (reference &&
        std::memcmp(triton_output.data(), reference_output.data(),
                    n * sizeof(float)) != 0) {
      throw std::runtime_error("generated reference output is not bit-exact");
    }
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
      mismatches += abs_error > 2.0e-6 + 2.0e-6 * std::abs(expected);
    }
    if (mismatches != 0) {
      throw std::runtime_error("Triton/KleidiAI output mismatch count=" +
                               std::to_string(mismatches) +
                               " max_abs=" + std::to_string(max_abs_error) +
                               " max_rel=" + std::to_string(max_rel_error));
    }

    const auto [triton_us, kleidiai_us] = paired_medians_us(
        run_triton, run_kleidiai, 100, iterations, batches);
    const auto [direct_us, wrapper_us] = paired_medians_us(
        run_triton, run_wrapper, 100, iterations, batches);
    std::cout << std::setprecision(8)
              << "PASS exact-KAI-layout W8 K=" << k << " N=" << n
              << " BLOCK_N=" << block_n << " requested=" << mode
              << " mode=" << selected_mode << '\n'
              << "triton_kernel_us=" << triton_us << '\n'
              << "kleidiai_kernel_us=" << kleidiai_us << '\n'
              << "triton_over_kleidiai=" << triton_us / kleidiai_us << "x\n"
              << "wrapper_pair_direct_us=" << direct_us << '\n'
              << "libtriton_jit_us=" << wrapper_us << '\n'
              << "libtriton_jit_minus_direct_us=" << wrapper_us - direct_us
              << '\n'
              << "libtriton_jit_over_kleidiai=" << wrapper_us / kleidiai_us
              << "x\n"
              << "max_abs_error=" << max_abs_error << '\n'
              << "max_rel_error=" << max_rel_error << '\n'
              << "pack_excluded=true\n";
    if (reference) {
      const auto [reference_us, active_us] = paired_medians_us(
          run_reference, run_triton, 100, iterations, batches);
      std::cout << "reference_triton_us=" << reference_us << '\n'
                << "active_triton_us=" << active_us << '\n'
                << "active_speedup=" << reference_us / active_us << "x\n"
                << "paired_bit_exact=true\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
