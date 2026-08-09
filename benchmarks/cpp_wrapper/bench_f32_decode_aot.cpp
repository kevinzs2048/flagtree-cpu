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

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {
using TritonKernel = void (*)(void*, void*, void*, void*, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t);
using QuantKernel = float (*)(const float*, int8_t*, int64_t);
using RuntimeKernel = void (*)(const int8_t*, float, const int8_t*,
                              const float*, float*, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t);

struct SharedObject {
  explicit SharedObject(const std::string& path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(dlerror());
  }
  ~SharedObject() { if (handle) dlclose(handle); }
  template <typename Function> Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle, name);
    if (const char* error = dlerror()) throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }
  void* handle = nullptr;
};

template <typename Function>
double median_us(Function&& function, int warmup, int iterations, int batches) {
  for (int i = 0; i < warmup; ++i) function();
  std::vector<double> samples;
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

std::string dirname(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::vector<int8_t> pack_blocked(const std::vector<int8_t>& weight,
                                 int k, int n, int block_n) {
  const int k4 = k / 4;
  const int n4 = n / 4;
  const int groups = block_n / 4;
  const int blocks = n / block_n;
  std::vector<int8_t> packed(static_cast<size_t>(k) * n);
  for (int block = 0; block < blocks; ++block) {
    for (int kb = 0; kb < k4; ++kb) {
      for (int group = 0; group < groups; ++group) {
        int8_t* tile =
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

int main(int argc, char** argv) {
  if (argc < 7 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " KERNEL_SO RUNTIME_SO K N BLOCK_N ITERS [BATCHES]\n";
    return 2;
  }
  try {
    const std::string kernel_path = argv[1];
    const std::string runtime_path = argv[2];
    const int k = std::stoi(argv[3]);
    const int n = std::stoi(argv[4]);
    const int block_n = std::stoi(argv[5]);
    const int iterations = std::stoi(argv[6]);
    const int batches = argc == 8 ? std::stoi(argv[7]) : 9;
    const int grid_x = n / block_n;
    std::vector<float> x(k);
    std::vector<int8_t> weight(static_cast<size_t>(k) * n);
    std::vector<float> scale(n);
    for (int inner = 0; inner < k; ++inner) {
      x[inner] = std::sin(inner * 0.017f) * 1.75f;
    }
    for (size_t i = 0; i < weight.size(); ++i) {
      weight[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
    }
    for (int col = 0; col < n; ++col) {
      scale[col] = 0.001f + (col % 31) * 0.00003f;
    }
    std::vector<int8_t> packed =
        pack_blocked(weight, k, n, block_n);
    std::vector<int8_t> xq(k);
    std::vector<float> direct_output(n);
    std::vector<float> wrapper_output(n);
    std::vector<float> runtime_output(n);

    SharedObject kernel_library(kernel_path);
    TritonKernel direct =
        kernel_library.symbol<TritonKernel>("_f32_w8_gemv_kernel");
    SharedObject runtime_library(runtime_path);
    QuantKernel quant =
        runtime_library.symbol<QuantKernel>("sdot_quant_act_f32");
    RuntimeKernel runtime = runtime_library.symbol<RuntimeKernel>(
        "sdot_gemv_blk_prequant_f32_range");
    auto run_direct = [&] {
      for (uint32_t pid = 0; pid < static_cast<uint32_t>(grid_x); ++pid) {
        direct(x.data(), packed.data(), scale.data(), direct_output.data(),
               pid, 0, 0, grid_x, 1, 1);
      }
    };
    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> wrapper(
        dirname(kernel_path), "_f32_w8_gemv_kernel");
    void* x_ptr = x.data();
    void* packed_ptr = packed.data();
    void* scale_ptr = scale.data();
    void* output_ptr = wrapper_output.data();
    void* args[] = {&x_ptr, &packed_ptr, &scale_ptr, &output_ptr};
    auto run_wrapper = [&] {
      wrapper.launch_with_signature(
          grid_x, 1, 1, 1, nullptr, args, "*fp32,*i8,*fp32,*fp32", 4);
    };
    auto run_runtime = [&] {
      const float x_scale = quant(x.data(), xq.data(), k);
      runtime(xq.data(), x_scale, packed.data(), scale.data(),
              runtime_output.data(), k, n, n / 4, block_n / 4, 0, grid_x);
    };
    run_direct();
    run_wrapper();
    run_runtime();
    if (direct_output != wrapper_output || direct_output != runtime_output) {
      throw std::runtime_error("FP32 output mismatch");
    }
    constexpr int warmup = 100;
    const double direct_us =
        median_us(run_direct, warmup, iterations, batches);
    const double wrapper_us =
        median_us(run_wrapper, warmup, iterations, batches);
    const double runtime_us =
        median_us(run_runtime, warmup, iterations, batches);
    std::cout << "PASS K=" << k << " N=" << n
              << " BLOCK_N=" << block_n << " GRID_X=" << grid_x << '\n'
              << "direct_aot_triton_us=" << direct_us << '\n'
              << "libtriton_jit_wrapper_us=" << wrapper_us << '\n'
              << "ggml_c_runtime_us=" << runtime_us << '\n'
              << "wrapper_minus_direct_us=" << wrapper_us - direct_us << '\n'
              << "wrapper_over_c=" << wrapper_us / runtime_us << "x\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
