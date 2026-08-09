#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {

using TritonKernel = void (*)(void*,
                              void*,
                              void*,
                              uint32_t,
                              uint32_t,
                              uint32_t,
                              uint32_t,
                              uint32_t,
                              uint32_t);
using CKernel = void (*)(const int8_t*, const int8_t*, int32_t*, int, int, int);

struct SharedObject {
  explicit SharedObject(const std::string& path) {
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
  Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle, name);
    if (const char* error = dlerror()) {
      throw std::runtime_error(error);
    }
    return reinterpret_cast<Function>(address);
  }

  void* handle = nullptr;
};

std::string dirname(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

template <typename Function>
double median_us(Function&& function, int warmup, int iterations, int batches) {
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
    const double nanoseconds =
        std::chrono::duration<double, std::nano>(end - begin).count();
    samples.push_back(nanoseconds / iterations / 1000.0);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void check(const std::vector<int8_t>& a,
           const std::vector<int8_t>& b,
           const std::vector<int32_t>& actual,
           int m,
           int n,
           int k,
           const char* label) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int32_t expected = 0;
      for (int inner = 0; inner < k; ++inner) {
        expected += static_cast<int32_t>(a[row * k + inner]) *
                    static_cast<int32_t>(b[inner * n + col]);
      }
      if (actual[row * n + col] != expected) {
        throw std::runtime_error(std::string(label) +
                                 " produced an incorrect result");
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 9) {
    std::cerr << "usage: " << argv[0]
              << " KERNEL_SO C_REFERENCE_SO K [GRID_X] [GRID_Y] [ITERS]"
                 " [BLOCK_M] [BLOCK_N]\n";
    return 2;
  }

  try {
    const std::string kernel_path = argv[1];
    const std::string c_path = argv[2];
    const int k = std::stoi(argv[3]);
    const uint32_t grid_x = argc > 4 ? std::stoul(argv[4]) : 1;
    const uint32_t grid_y = argc > 5 ? std::stoul(argv[5]) : 1;
    const int iterations = argc > 6 ? std::stoi(argv[6]) : 10000;
    const int block_m = argc > 7 ? std::stoi(argv[7]) : 32;
    const int block_n = argc > 8 ? std::stoi(argv[8]) : 32;
    const int m = static_cast<int>(grid_x) * block_m;
    const int n = static_cast<int>(grid_y) * block_n;

    std::vector<int8_t> a(static_cast<size_t>(m) * k);
    std::vector<int8_t> b(static_cast<size_t>(k) * n);
    std::vector<int32_t> triton_output(static_cast<size_t>(m) * n);
    std::vector<int32_t> c_output(static_cast<size_t>(m) * n);
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = static_cast<int8_t>((i * 17 + 13) % 255 - 127);
    }
    for (size_t i = 0; i < b.size(); ++i) {
      b[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
    }

    SharedObject direct_library(kernel_path);
    TritonKernel direct =
        direct_library.symbol<TritonKernel>("i8mm_kernel");
    SharedObject c_library(c_path);
    CKernel c_kernel = c_library.symbol<CKernel>("gemm_sve2_i8mm");

    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> wrapper(
        dirname(kernel_path), "i8mm_kernel");
    void* a_pointer = a.data();
    void* b_pointer = b.data();
    void* wrapper_c_pointer = triton_output.data();
    void* wrapper_args[] = {
        &a_pointer, &b_pointer, &wrapper_c_pointer};

    auto run_direct = [&] {
      for (uint32_t z = 0; z < 1; ++z) {
        for (uint32_t y = 0; y < grid_y; ++y) {
          for (uint32_t x = 0; x < grid_x; ++x) {
            direct(a.data(),
                   b.data(),
                   triton_output.data(),
                   x,
                   y,
                   z,
                   grid_x,
                   grid_y,
                   1);
          }
        }
      }
    };
    auto run_wrapper = [&] {
      wrapper.launch_with_signature(
          grid_x,
          grid_y,
          1,
          1,
          nullptr,
          wrapper_args,
          "*i8,*i8,*i32",
          3);
    };
    auto run_c = [&] {
      c_kernel(a.data(), b.data(), c_output.data(), m, n, k);
    };

    run_direct();
    run_wrapper();
    run_c();
    check(a, b, triton_output, m, n, k, "generated Triton kernel");
    check(a, b, c_output, m, n, k, "ACLE C");

    const int warmup = 200;
    const int batches = 9;
    const double direct_us =
        median_us(run_direct, warmup, iterations, batches);
    const double wrapper_us =
        median_us(run_wrapper, warmup, iterations, batches);
    const double c_us = median_us(run_c, warmup, iterations, batches);

    std::cout << "PASS M=" << m << " N=" << n << " K=" << k << '\n'
              << "direct_triton_us=" << direct_us << '\n'
              << "libtriton_jit_cpu_wrapper_us=" << wrapper_us << '\n'
              << "wrapper_minus_direct_us=" << wrapper_us - direct_us << '\n'
              << "optimized_c_us=" << c_us << '\n'
              << "wrapper_over_c=" << wrapper_us / c_us << "x\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
