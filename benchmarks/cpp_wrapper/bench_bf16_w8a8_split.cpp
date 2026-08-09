#include <dlfcn.h>

#include <algorithm>
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

namespace {

using QuantKernel = void (*)(void *, void *, void *, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, uint32_t);
using GemvKernel = void (*)(void *, void *, void *, void *, void *, int32_t,
                            int32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t);
using CKernel = void (*)(const uint16_t *, int8_t *, float *, const int8_t *,
                         const float *, uint16_t *, int, int);
using WideCKernel = void (*)(const uint16_t *, int8_t *, float *,
                             const int8_t *, const float *, uint16_t *, int,
                             int, int);

uint16_t float_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

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
  if (argc < 7 || argc > 10) {
    std::cerr << "usage: " << argv[0]
              << " QUANT_SO GEMV_SO C_SO K N ITERS [BATCHES] [BLOCK_N] "
                 "[REFERENCE_QUANT_SO]\n";
    return 2;
  }

  try {
    const int k = std::stoi(argv[4]);
    const int n = std::stoi(argv[5]);
    const int iterations = std::stoi(argv[6]);
    const int batches = argc >= 8 ? std::stoi(argv[7]) : 9;
    const int block_n = argc >= 9 ? std::stoi(argv[8]) : 4;
    if (k <= 0 || n <= 0 || k % 4 != 0 || n % block_n != 0 ||
        block_n % 4 != 0) {
      throw std::runtime_error("invalid K/N/BLOCK_N divisibility");
    }

    const int k_groups = k / 4;
    const int tiles = n / 4;
    std::vector<uint16_t> x(k);
    std::vector<int8_t> triton_xq(k);
    std::vector<int8_t> reference_xq(k);
    std::vector<int8_t> c_xq(k);
    std::vector<float> triton_x_scale(1);
    std::vector<float> reference_x_scale(1);
    std::vector<float> c_x_scale(1);
    std::vector<int8_t> packed(static_cast<size_t>(tiles) * k_groups * 16);
    std::vector<float> weight_scale(n);
    std::vector<uint16_t> triton_output(n);
    std::vector<uint16_t> reference_output(n);
    std::vector<uint16_t> c_output(n);

    for (int i = 0; i < k; ++i) {
      x[i] = float_to_bf16(std::sin(i * 0.017f) * 1.75f);
    }
    for (size_t i = 0; i < packed.size(); ++i) {
      packed[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
    }
    for (int i = 0; i < n; ++i) {
      weight_scale[i] = 0.002f + (i % 29) * 0.00003f;
    }

    SharedObject quant_library(argv[1]);
    SharedObject gemv_library(argv[2]);
    SharedObject c_library(argv[3]);
    QuantKernel quant =
        quant_library.symbol<QuantKernel>("_quantize_bf16_w8_kernel");
    std::unique_ptr<SharedObject> reference_quant_library;
    QuantKernel reference_quant = nullptr;
    if (argc == 10) {
      reference_quant_library = std::make_unique<SharedObject>(argv[9]);
      reference_quant = reference_quant_library->symbol<QuantKernel>(
          "_quantize_bf16_w8_kernel");
    }
    const bool wide = block_n != 4;
    GemvKernel gemv = gemv_library.symbol<GemvKernel>(
        wide ? "_w8a8_wide_gemv_kernel" : "_w8a8_grouped_gemv_kernel");
    CKernel c_kernel = nullptr;
    WideCKernel wide_c_kernel = nullptr;
    if (wide) {
      wide_c_kernel = c_library.symbol<WideCKernel>("bf16_w8a8_wide_c");
    } else {
      c_kernel = c_library.symbol<CKernel>("bf16_w8a8_split_c");
    }

    auto run_quant = [&] {
      quant(x.data(), triton_xq.data(), triton_x_scale.data(), 0, 0, 0, 1, 1,
            1);
    };
    auto run_gemv = [&] {
      gemv(triton_xq.data(), triton_x_scale.data(), packed.data(),
           weight_scale.data(), triton_output.data(), 0,
           wide ? n / block_n : tiles, 0, 0, 0, 1, 1, 1);
    };
    auto run_triton = [&] {
      run_quant();
      run_gemv();
    };
    auto run_reference = [&] {
      reference_quant(x.data(), reference_xq.data(), reference_x_scale.data(),
                      0, 0, 0, 1, 1, 1);
      gemv(reference_xq.data(), reference_x_scale.data(), packed.data(),
           weight_scale.data(), reference_output.data(), 0,
           wide ? n / block_n : tiles, 0, 0, 0, 1, 1, 1);
    };
    auto run_c = [&] {
      if (wide) {
        wide_c_kernel(x.data(), c_xq.data(), c_x_scale.data(), packed.data(),
                      weight_scale.data(), c_output.data(), k, n, block_n);
      } else {
        c_kernel(x.data(), c_xq.data(), c_x_scale.data(), packed.data(),
                 weight_scale.data(), c_output.data(), k, n);
      }
    };

    run_triton();
    if (reference_quant)
      run_reference();
    run_c();
    if (triton_xq != c_xq || triton_output != c_output ||
        triton_x_scale[0] != c_x_scale[0]) {
      size_t output_mismatches = 0;
      for (int i = 0; i < n; ++i) {
        output_mismatches += triton_output[i] != c_output[i];
      }
      throw std::runtime_error(
          "output mismatch: xq=" + std::to_string(triton_xq != c_xq) +
          " scale=" + std::to_string(triton_x_scale[0] != c_x_scale[0]) +
          " output=" + std::to_string(output_mismatches));
    }
    if (reference_quant &&
        (triton_xq != reference_xq || triton_x_scale != reference_x_scale ||
         triton_output != reference_output)) {
      throw std::runtime_error("reference generated pipeline mismatch");
    }

    const double quant_us = median_us(run_quant, 100, iterations, batches);
    const double gemv_us = median_us(run_gemv, 100, iterations, batches);
    const double triton_us = median_us(run_triton, 100, iterations, batches);
    const double c_us = median_us(run_c, 100, iterations, batches);
    std::cout << "PASS BF16-W8 K=" << k << " N=" << n << '\n'
              << "direct_quant_triton_us=" << quant_us << '\n'
              << "direct_gemv_triton_us=" << gemv_us << '\n'
              << "split_sum_triton_us=" << quant_us + gemv_us << '\n'
              << "direct_split_triton_us=" << triton_us << '\n'
              << "fused_acle_c_us=" << c_us << '\n'
              << "triton_over_c=" << triton_us / c_us << "x\n"
              << "bit_exact=true\n";
    if (reference_quant) {
      const auto [reference_us, active_us] =
          paired_median_us(run_reference, run_triton, 100, iterations, batches);
      std::cout << "reference_pipeline_us=" << reference_us << '\n'
                << "active_paired_pipeline_us=" << active_us << '\n'
                << "active_speedup=" << reference_us / active_us << "x\n"
                << "paired_bit_exact=true\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
