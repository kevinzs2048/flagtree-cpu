#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {

using Probe = void (*)(void*,
                       void*,
                       void*,
                       uint32_t,
                       uint32_t,
                       uint32_t,
                       uint32_t,
                       uint32_t,
                       uint32_t);

template <typename Function>
double median_ns(Function&& function, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    function();
  }
  std::vector<double> samples;
  for (int batch = 0; batch < 9; ++batch) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      function();
    }
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::nano>(end - begin).count() /
        iterations);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::string dirname(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: " << argv[0] << " PROBE_SO [ITERS]\n";
    return 2;
  }

  try {
    const std::string path = argv[1];
    const int iterations = argc == 3 ? std::stoi(argv[2]) : 5000000;
    void* module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) {
      throw std::runtime_error(dlerror());
    }
    auto direct =
        reinterpret_cast<Probe>(dlsym(module, "three_pointer_probe"));
    if (!direct) {
      throw std::runtime_error(dlerror());
    }

    int8_t a = 1;
    int8_t b = 1;
    int32_t direct_c = 0;
    int32_t wrapper_c = 0;
    void* a_pointer = &a;
    void* b_pointer = &b;
    void* c_pointer = &wrapper_c;
    void* args[] = {&a_pointer, &b_pointer, &c_pointer};

    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> wrapper(
        dirname(path), "three_pointer_probe");
    auto run_direct = [&] {
      direct(&a, &b, &direct_c, 0, 0, 0, 1, 1, 1);
    };
    auto run_wrapper = [&] {
      wrapper.launch_with_signature(
          1, 1, 1, 1, nullptr, args, "*i8,*i8,*i32", 3);
    };

    run_direct();
    run_wrapper();
    if (direct_c != 2 || wrapper_c != 2) {
      throw std::runtime_error("probe produced an incorrect result");
    }

    const double direct_ns = median_ns(run_direct, 10000, iterations);
    const double wrapper_ns = median_ns(run_wrapper, 10000, iterations);
    std::cout << "PASS probe\n"
              << "direct_function_ns=" << direct_ns << '\n'
              << "libtriton_jit_cpu_wrapper_ns=" << wrapper_ns << '\n'
              << "wrapper_minus_direct_ns=" << wrapper_ns - direct_ns << '\n';
    dlclose(module);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
