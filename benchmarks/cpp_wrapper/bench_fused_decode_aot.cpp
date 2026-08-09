#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {

using TritonKernel = void (*)(void *, void *, void *, void *, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
using RuntimeKernel = void (*)(const uint16_t *, const int8_t *, const float *,
                               uint16_t *, int64_t, int64_t, int64_t);

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

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float from_bf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

template <typename Function>
double median_us(Function &&function, int warmup, int iterations, int batches) {
  for (int i = 0; i < warmup; ++i) {
    function();
  }
  std::vector<double> samples;
  samples.reserve(batches);
  for (int batch = 0; batch < batches; ++batch) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      function();
    }
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count() /
        iterations);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

struct PairedLatency {
  double first;
  double second;
};

template <typename First, typename Second>
PairedLatency paired_median_us(First &&first, Second &&second, int warmup,
                               int iterations, int batches) {
  for (int i = 0; i < warmup; ++i) {
    first();
    second();
  }
  std::vector<double> first_samples;
  std::vector<double> second_samples;
  first_samples.reserve(batches);
  second_samples.reserve(batches);
  auto measure = [iterations](auto &&function) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      function();
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count() /
           iterations;
  };
  for (int batch = 0; batch < batches; ++batch) {
    if ((batch & 1) == 0) {
      first_samples.push_back(measure(first));
      second_samples.push_back(measure(second));
    } else {
      second_samples.push_back(measure(second));
      first_samples.push_back(measure(first));
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

void check_codegen(const std::vector<uint16_t> &x,
                   const std::vector<int8_t> &weight,
                   const std::vector<float> &weight_scale,
                   const std::vector<uint16_t> &output, int k, int n) {
  float absmax = 0.0f;
  for (uint16_t value : x) {
    absmax = std::max(absmax, std::abs(from_bf16(value)));
  }
  absmax = std::max(absmax, 1.0e-8f);
  const float inv_scale = 127.0f / absmax;
  const float x_scale = absmax / 127.0f;
  std::vector<int8_t> quantized(k);
  for (int inner = 0; inner < k; ++inner) {
    int value =
        static_cast<int>(std::nearbyint(from_bf16(x[inner]) * inv_scale));
    value = std::clamp(value, -128, 127);
    quantized[inner] = static_cast<int8_t>(value);
  }
  for (int col = 0; col < n; ++col) {
    int32_t acc = 0;
    for (int inner = 0; inner < k; ++inner) {
      acc += static_cast<int32_t>(quantized[inner]) *
             static_cast<int32_t>(weight[inner * n + col]);
    }
    const uint16_t expected =
        to_bf16(static_cast<float>(acc) * x_scale * weight_scale[col]);
    if (output[col] != expected) {
      throw std::runtime_error(
          "generated Triton fused decode produced an incorrect result");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 7 || argc > 10) {
    std::cerr << "usage: " << argv[0]
              << " KERNEL_SO RUNTIME_SO K N BLOCK_N ITERS [BATCHES] "
                 "[KERNEL_SYMBOL] [REFERENCE_KERNEL_SO]\n";
    return 2;
  }
  try {
    const std::string kernel_path = argv[1];
    const std::string runtime_path = argv[2];
    const int k = std::stoi(argv[3]);
    const int n = std::stoi(argv[4]);
    const int block_n = std::stoi(argv[5]);
    const int iterations = std::stoi(argv[6]);
    const int batches = argc >= 8 ? std::stoi(argv[7]) : 9;
    const std::string kernel_symbol =
        argc >= 9 ? argv[8] : "_tle_fused_bf16_gemv_kernel";
    const bool whole_microtile = kernel_symbol == "_tle_whole_bf16_gemv_kernel";
    if (k % 4 != 0 || n % block_n != 0 || block_n % 4 != 0) {
      throw std::runtime_error("invalid K/N/BLOCK_N divisibility");
    }
    const int k4 = k / 4;
    const int n4 = n / 4;
    const int groups = block_n / 4;
    const int grid_x = n / block_n;

    std::vector<uint16_t> x(k);
    std::vector<int8_t> weight(static_cast<size_t>(k) * n);
    std::vector<float> weight_scale(n);
    for (int inner = 0; inner < k; ++inner) {
      x[inner] = to_bf16(std::sin(static_cast<float>(inner) * 0.017f) * 1.75f);
    }
    for (size_t i = 0; i < weight.size(); ++i) {
      weight[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
    }
    for (int col = 0; col < n; ++col) {
      weight_scale[col] = 0.001f + (col % 31) * 0.00003f;
    }

    std::vector<int8_t> packed_kmajor(static_cast<size_t>(k) * n);
    for (int kb = 0; kb < k4; ++kb) {
      for (int nb = 0; nb < n4; ++nb) {
        int8_t *tile =
            packed_kmajor.data() + (static_cast<size_t>(kb) * n4 + nb) * 16;
        for (int ni = 0; ni < 4; ++ni) {
          for (int ki = 0; ki < 4; ++ki) {
            tile[ni * 4 + ki] = weight[(kb * 4 + ki) * n + nb * 4 + ni];
          }
        }
      }
    }
    std::vector<int8_t> packed_blocked(packed_kmajor.size());
    for (int block = 0; block < grid_x; ++block) {
      for (int kb = 0; kb < k4; ++kb) {
        for (int group = 0; group < groups; ++group) {
          const size_t source =
              (static_cast<size_t>(kb) * n4 + block * groups + group) * 16;
          const size_t destination =
              ((static_cast<size_t>(block) * k4 + kb) * groups + group) * 16;
          std::memcpy(packed_blocked.data() + destination,
                      packed_kmajor.data() + source, 16);
        }
      }
    }

    std::vector<uint16_t> direct_output(n);
    std::vector<uint16_t> reference_output(n);
    std::vector<uint16_t> wrapper_output(n);
    std::vector<uint16_t> runtime_output(n);
    SharedObject kernel_library(kernel_path);
    TritonKernel direct =
        kernel_library.symbol<TritonKernel>(kernel_symbol.c_str());
    std::unique_ptr<SharedObject> reference_library;
    TritonKernel reference = nullptr;
    if (argc == 10) {
      reference_library = std::make_unique<SharedObject>(argv[9]);
      reference =
          reference_library->symbol<TritonKernel>(kernel_symbol.c_str());
    }
    SharedObject runtime_library(runtime_path);
    RuntimeKernel runtime =
        runtime_library.symbol<RuntimeKernel>("sdot_gemv_m1_fused_bf16");

    auto run_direct = [&] {
      const uint32_t launch_grid_x = whole_microtile ? 1 : grid_x;
      for (uint32_t pid = 0; pid < launch_grid_x; ++pid) {
        direct(x.data(), packed_blocked.data(), weight_scale.data(),
               direct_output.data(), pid, 0, 0, launch_grid_x, 1, 1);
      }
    };
    auto run_reference = [&] {
      const uint32_t launch_grid_x = whole_microtile ? 1 : grid_x;
      for (uint32_t pid = 0; pid < launch_grid_x; ++pid) {
        reference(x.data(), packed_blocked.data(), weight_scale.data(),
                  reference_output.data(), pid, 0, 0, launch_grid_x, 1, 1);
      }
    };

    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> wrapper(
        dirname(kernel_path), kernel_symbol);
    void *x_ptr = x.data();
    void *packed_ptr = packed_blocked.data();
    void *scale_ptr = weight_scale.data();
    void *output_ptr = wrapper_output.data();
    void *wrapper_args[] = {&x_ptr, &packed_ptr, &scale_ptr, &output_ptr};
    auto run_wrapper = [&] {
      wrapper.launch_with_signature(whole_microtile ? 1 : grid_x, 1, 1, 1,
                                    nullptr, wrapper_args,
                                    "*bf16,*i8,*fp32,*bf16", 4);
    };
    auto run_runtime = [&] {
      runtime(x.data(), packed_kmajor.data(), weight_scale.data(),
              runtime_output.data(), k, n, n4);
    };

    run_direct();
    if (reference) {
      run_reference();
    }
    run_wrapper();
    run_runtime();
    check_codegen(x, weight, weight_scale, direct_output, k, n);
    if (reference && direct_output != reference_output) {
      throw std::runtime_error("reference generated output mismatch");
    }
    if (direct_output != wrapper_output) {
      throw std::runtime_error("libtriton-jit wrapper output mismatch");
    }

    constexpr int warmup = 100;
    const double direct_us = median_us(run_direct, warmup, iterations, batches);
    const double wrapper_us =
        median_us(run_wrapper, warmup, iterations, batches);
    const double runtime_us =
        median_us(run_runtime, warmup, iterations, batches);
    std::cout << "PASS K=" << k << " N=" << n
              << (whole_microtile ? " TILE_N=" : " BLOCK_N=") << block_n
              << " GRID_X=" << (whole_microtile ? 1 : grid_x) << '\n'
              << "direct_aot_triton_us=" << direct_us << '\n'
              << "libtriton_jit_wrapper_us=" << wrapper_us << '\n'
              << "legacy_c_runtime_us=" << runtime_us << '\n'
              << "wrapper_minus_direct_us=" << wrapper_us - direct_us << '\n'
              << "wrapper_over_c=" << wrapper_us / runtime_us << "x\n";
    if (reference) {
      const auto [reference_us, active_us] = paired_median_us(
          run_reference, run_direct, warmup, iterations, batches);
      std::cout << "reference_direct_us=" << reference_us << '\n'
                << "active_paired_direct_us=" << active_us << '\n'
                << "active_speedup=" << reference_us / active_us << "x\n"
                << "paired_bit_exact=true\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
