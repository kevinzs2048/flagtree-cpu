#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using TritonKernel = void (*)(void *, void *, void *, void *, void *,
                              int32_t, int32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t);
using CKernel = void (*)(const int8_t *, const float *, const uint8_t *,
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
  template <typename Function>
  Function symbol(const char *name) {
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

} // namespace

int main(int argc, char **argv) {
  if (argc < 6 || argc > 7) {
    std::cerr << "usage: " << argv[0]
              << " KERNEL_SO C_CEILING_SO K N ITERS [BATCHES]\n";
    return 2;
  }

  try {
    const std::string kernel_path = argv[1];
    const std::string c_path = argv[2];
    const int k = std::stoi(argv[3]);
    const int n = std::stoi(argv[4]);
    const int iterations = std::stoi(argv[5]);
    const int batches = argc == 7 ? std::stoi(argv[6]) : 9;
    if (k <= 0 || n <= 0 || k % 32 != 0 || n % 4 != 0) {
      throw std::runtime_error("K must be a multiple of 32 and N of 4");
    }

    const int groups = k / 32;
    const int tiles = n / 4;
    std::vector<int8_t> x(k);
    std::vector<float> x_scale(groups);
    std::vector<uint8_t> packed(
        static_cast<size_t>(tiles) * groups * 4 * 16);
    std::vector<float> weight_scale(
        static_cast<size_t>(tiles) * groups * 4);
    std::vector<float> triton_output(n);
    std::vector<float> c_output(n);

    for (int i = 0; i < k; ++i) {
      x[i] = static_cast<int8_t>((i * 17 + 13) % 255 - 127);
    }
    for (int group = 0; group < groups; ++group) {
      x_scale[group] = 0.001f + (group % 17) * 0.00007f;
    }
    for (size_t i = 0; i < packed.size(); ++i) {
      const uint8_t low = static_cast<uint8_t>((i * 7 + 3) & 15);
      const uint8_t high = static_cast<uint8_t>((i * 11 + 5) & 15);
      packed[i] = static_cast<uint8_t>(low | (high << 4));
    }
    for (size_t i = 0; i < weight_scale.size(); ++i) {
      weight_scale[i] = 0.002f + (i % 29) * 0.00003f;
    }

    SharedObject kernel_library(kernel_path);
    SharedObject c_library(c_path);
    TritonKernel kernel = kernel_library.symbol<TritonKernel>(
        "_w4a8_grouped_gemv_kernel");
    CKernel c_kernel =
        c_library.symbol<CKernel>("w4a8_microtile4_c");

    auto run_triton = [&] {
      kernel(x.data(), x_scale.data(), packed.data(), weight_scale.data(),
             triton_output.data(), 0, tiles, 0, 0, 0, 1, 1, 1);
    };
    auto run_c = [&] {
      c_kernel(x.data(), x_scale.data(), packed.data(), weight_scale.data(),
               c_output.data(), k, n);
    };

    run_triton();
    run_c();
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
      max_abs = std::max(max_abs,
                         std::abs(triton_output[i] - c_output[i]));
    }
    if (max_abs > 2.0e-5f) {
      throw std::runtime_error("output mismatch, max_abs=" +
                               std::to_string(max_abs));
    }

    const double triton_us =
        median_us(run_triton, 100, iterations, batches);
    const double c_us = median_us(run_c, 100, iterations, batches);
    std::cout << "PASS K=" << k << " N=" << n << '\n'
              << "direct_triton_us=" << triton_us << '\n'
              << "acle_c_us=" << c_us << '\n'
              << "triton_over_c=" << triton_us / c_us << "x\n"
              << "max_abs=" << max_abs << '\n';
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
