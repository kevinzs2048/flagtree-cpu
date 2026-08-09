#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using TritonKernel = void (*)(void *, void *, void *, void *, void *, int32_t,
                              int32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t);
using CKernel = void (*)(const int8_t *, const float *, const int8_t *,
                         const float *, float *, int, int);

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
    for (int i = 0; i < iterations; ++i)
      function();
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

} // namespace

int main(int argc, char **argv) {
  if (argc < 6 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " KERNEL_SO C_CEILING_SO K N ITERS [BATCHES] "
                 "[REFERENCE_KERNEL_SO]\n";
    return 2;
  }

  try {
    const std::string kernel_path = argv[1];
    const std::string c_path = argv[2];
    const int k = std::stoi(argv[3]);
    const int n = std::stoi(argv[4]);
    const int iterations = std::stoi(argv[5]);
    const int batches = argc >= 7 ? std::stoi(argv[6]) : 9;
    if (k <= 0 || n <= 0 || k % 4 != 0 || n % 4 != 0) {
      throw std::runtime_error("K and N must be positive multiples of 4");
    }

    const int k_groups = k / 4;
    const int tiles = n / 4;
    std::vector<int8_t> x(k);
    std::vector<float> x_scale(1);
    std::vector<int8_t> packed(static_cast<size_t>(tiles) * k_groups * 16);
    std::vector<float> weight_scale(n);
    std::vector<float> triton_output(n);
    std::vector<float> reference_output(n);
    std::vector<float> c_output(n);

    for (int i = 0; i < k; ++i) {
      x[i] = static_cast<int8_t>((i * 17 + 13) % 255 - 127);
    }
    x_scale[0] = 0.0037f;
    for (size_t i = 0; i < packed.size(); ++i) {
      packed[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
    }
    for (int i = 0; i < n; ++i) {
      weight_scale[i] = 0.002f + (i % 29) * 0.00003f;
    }

    SharedObject kernel_library(kernel_path);
    SharedObject c_library(c_path);
    TritonKernel kernel =
        kernel_library.symbol<TritonKernel>("_w8a8_grouped_gemv_kernel");
    std::unique_ptr<SharedObject> reference_library;
    TritonKernel reference = nullptr;
    if (argc == 8) {
      reference_library = std::make_unique<SharedObject>(argv[7]);
      reference =
          reference_library->symbol<TritonKernel>("_w8a8_grouped_gemv_kernel");
    }
    CKernel c_kernel = c_library.symbol<CKernel>("w8a8_microtile4_c");

    auto run_triton = [&] {
      kernel(x.data(), x_scale.data(), packed.data(), weight_scale.data(),
             triton_output.data(), 0, tiles, 0, 0, 0, 1, 1, 1);
    };
    auto run_c = [&] {
      c_kernel(x.data(), x_scale.data(), packed.data(), weight_scale.data(),
               c_output.data(), k, n);
    };
    auto run_reference = [&] {
      reference(x.data(), x_scale.data(), packed.data(), weight_scale.data(),
                reference_output.data(), 0, tiles, 0, 0, 0, 1, 1, 1);
    };

    run_triton();
    if (reference)
      run_reference();
    run_c();
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
      max_abs = std::max(max_abs, std::abs(triton_output[i] - c_output[i]));
    }
    if (max_abs > 2.0e-5f) {
      throw std::runtime_error("output mismatch, max_abs=" +
                               std::to_string(max_abs));
    }
    if (reference && triton_output != reference_output) {
      throw std::runtime_error("reference generated output mismatch");
    }

    const double triton_us = median_us(run_triton, 100, iterations, batches);
    const double c_us = median_us(run_c, 100, iterations, batches);
    std::cout << "PASS W8A8 K=" << k << " N=" << n << '\n'
              << "direct_triton_us=" << triton_us << '\n'
              << "acle_c_us=" << c_us << '\n'
              << "triton_over_c=" << triton_us / c_us << "x\n"
              << "max_abs=" << max_abs << '\n';
    if (reference) {
      const auto [reference_us, active_us] =
          paired_median_us(run_reference, run_triton, 100, iterations, batches);
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
