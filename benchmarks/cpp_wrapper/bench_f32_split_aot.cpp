#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using QuantAotKernel = void (*)(void *, void *, void *, uint32_t, uint32_t,
                                uint32_t, uint32_t, uint32_t, uint32_t);
using GemvAotKernel = void (*)(void *, void *, void *, void *, void *,
                               uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t);
using QuantCKernel = float (*)(const float *, int8_t *, int64_t);
using GemvCKernel = void (*)(const int8_t *, float, const int8_t *,
                             const float *, float *, int64_t, int64_t,
                             int64_t, int64_t, int64_t, int64_t);

struct SharedObject {
  explicit SharedObject(const std::string & path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(dlerror());
  }
  ~SharedObject() {
    if (handle) dlclose(handle);
  }
  template <typename Function>
  Function symbol(const char * name) {
    dlerror();
    void * address = dlsym(handle, name);
    if (const char * error = dlerror()) throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }
  void * handle = nullptr;
};

template <typename Function>
double median_us(Function && function, int warmup, int iterations, int batches) {
  for (int i = 0; i < warmup; ++i) function();
  std::vector<double> samples;
  samples.reserve(batches);
  for (int batch = 0; batch < batches; ++batch) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) function();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count() /
        iterations);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::vector<int8_t> pack_blocked(const std::vector<int8_t> & weight,
                                 int k, int n, int block_n) {
  const int k4 = k / 4;
  const int groups = block_n / 4;
  const int blocks = n / block_n;
  std::vector<int8_t> packed(static_cast<size_t>(k) * n);
  for (int block = 0; block < blocks; ++block) {
    for (int kb = 0; kb < k4; ++kb) {
      for (int group = 0; group < groups; ++group) {
        int8_t * tile =
            packed.data() +
            ((static_cast<size_t>(block) * k4 + kb) * groups + group) * 16;
        for (int ni = 0; ni < 4; ++ni) {
          for (int ki = 0; ki < 4; ++ki) {
            const int col = block * block_n + group * 4 + ni;
            tile[ni * 4 + ki] = weight[(kb * 4 + ki) * n + col];
          }
        }
      }
    }
  }
  return packed;
}
}  // namespace

int main(int argc, char ** argv) {
  if (argc < 8 || argc > 9) {
    std::cerr << "usage: " << argv[0]
              << " QUANT_SO GEMV_SO RUNTIME_SO K N BLOCK_N ITERS [BATCHES]\n";
    return 2;
  }
  try {
    const std::string quant_path = argv[1];
    const std::string gemv_path = argv[2];
    const std::string runtime_path = argv[3];
    const int k = std::stoi(argv[4]);
    const int n = std::stoi(argv[5]);
    const int block_n = std::stoi(argv[6]);
    const int iterations = std::stoi(argv[7]);
    const int batches = argc == 9 ? std::stoi(argv[8]) : 9;
    if (k % 4 != 0 || n % block_n != 0 || block_n % 4 != 0) {
      throw std::runtime_error("invalid K/N/BLOCK_N divisibility");
    }
    const int grid_x = n / block_n;
    std::vector<float> x(k);
    std::vector<int8_t> weight(static_cast<size_t>(k) * n);
    std::vector<float> weight_scale(n);
    for (int inner = 0; inner < k; ++inner) {
      x[inner] = std::sin(inner * 0.017f) * 1.75f;
    }
    for (size_t i = 0; i < weight.size(); ++i) {
      weight[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
    }
    for (int col = 0; col < n; ++col) {
      weight_scale[col] = 0.001f + (col % 31) * 0.00003f;
    }
    std::vector<int8_t> packed = pack_blocked(weight, k, n, block_n);
    std::vector<int8_t> triton_xq(k);
    std::vector<int8_t> runtime_xq(k);
    std::vector<float> triton_x_scale(1);
    std::vector<float> triton_output(n);
    std::vector<float> runtime_output(n);

    SharedObject quant_library(quant_path);
    SharedObject gemv_library(gemv_path);
    SharedObject runtime_library(runtime_path);
    QuantAotKernel quant_aot = quant_library.symbol<QuantAotKernel>(
        "_quantize_f32_w8_kernel");
    GemvAotKernel gemv_aot = gemv_library.symbol<GemvAotKernel>(
        "_f32_w8_prequant_gemv_kernel");
    QuantCKernel quant_c =
        runtime_library.symbol<QuantCKernel>("sdot_quant_act_f32");
    GemvCKernel gemv_c = runtime_library.symbol<GemvCKernel>(
        "sdot_gemv_blk_prequant_f32_range");

    auto run_triton = [&] {
      quant_aot(x.data(), triton_xq.data(), triton_x_scale.data(),
                0, 0, 0, 1, 1, 1);
      for (uint32_t pid = 0; pid < static_cast<uint32_t>(grid_x); ++pid) {
        gemv_aot(triton_xq.data(), triton_x_scale.data(), packed.data(),
                 weight_scale.data(), triton_output.data(),
                 pid, 0, 0, grid_x, 1, 1);
      }
    };
    auto run_runtime = [&] {
      const float scale = quant_c(x.data(), runtime_xq.data(), k);
      gemv_c(runtime_xq.data(), scale, packed.data(), weight_scale.data(),
             runtime_output.data(), k, n, n / 4, block_n / 4, 0, grid_x);
    };
    run_triton();
    run_runtime();
    if (triton_xq != runtime_xq || triton_output != runtime_output) {
      throw std::runtime_error("split AOT output mismatch");
    }

    constexpr int warmup = 100;
    const double triton_us =
        median_us(run_triton, warmup, iterations, batches);
    const double runtime_us =
        median_us(run_runtime, warmup, iterations, batches);
    std::cout << "PASS K=" << k << " N=" << n
              << " BLOCK_N=" << block_n << " GRID_X=" << grid_x << '\n'
              << "split_direct_aot_triton_us=" << triton_us << '\n'
              << "ggml_c_runtime_us=" << runtime_us << '\n'
              << "triton_over_c=" << triton_us / runtime_us << "x\n";
  } catch (const std::exception & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
