#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {

using TritonKernel = void (*)(void *, void *, void *, void *, void *, void *,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t);
using RuntimeKernel = void (*)(const uint16_t *, const int8_t *, const int8_t *,
                               const float *, const float *, uint16_t *,
                               int64_t, int64_t);

struct SharedObject {
  explicit SharedObject(const std::string &path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
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
  for (int i = 0; i < warmup; ++i)
    function();
  std::vector<double> samples;
  samples.reserve(batches);
  for (int batch = 0; batch < batches; ++batch) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      function();
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

std::string dirname(const std::string &path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::vector<int8_t> pack_kmajor(const std::vector<int8_t> &weight, int k,
                                int n) {
  const int k4 = k / 4;
  const int n4 = n / 4;
  std::vector<int8_t> packed(static_cast<size_t>(k) * n);
  for (int kb = 0; kb < k4; ++kb) {
    for (int nb = 0; nb < n4; ++nb) {
      int8_t *tile = packed.data() + (static_cast<size_t>(kb) * n4 + nb) * 16;
      for (int ni = 0; ni < 4; ++ni) {
        for (int ki = 0; ki < 4; ++ki) {
          tile[ni * 4 + ki] = weight[(kb * 4 + ki) * n + nb * 4 + ni];
        }
      }
    }
  }
  return packed;
}

std::vector<int8_t> pack_blocked(const std::vector<int8_t> &source, int k,
                                 int n, int block_n) {
  const int k4 = k / 4;
  const int n4 = n / 4;
  const int groups = block_n / 4;
  const int grid_x = n / block_n;
  std::vector<int8_t> packed(source.size());
  for (int block = 0; block < grid_x; ++block) {
    for (int kb = 0; kb < k4; ++kb) {
      for (int group = 0; group < groups; ++group) {
        const size_t src =
            (static_cast<size_t>(kb) * n4 + block * groups + group) * 16;
        const size_t dst =
            ((static_cast<size_t>(block) * k4 + kb) * groups + group) * 16;
        std::memcpy(packed.data() + dst, source.data() + src, 16);
      }
    }
  }
  return packed;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 7 || argc > 9) {
    std::cerr << "usage: " << argv[0]
              << " KERNEL_SO RUNTIME_SO K N BLOCK_N ITERS [BATCHES] "
                 "[REFERENCE_KERNEL_SO]\n";
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
    if (k % 4 != 0 || n % block_n != 0 || block_n % 4 != 0) {
      throw std::runtime_error("invalid K/N/BLOCK_N divisibility");
    }
    const int grid_x = n / block_n;

    std::vector<uint16_t> x(k);
    std::vector<int8_t> gate_weight(static_cast<size_t>(k) * n);
    std::vector<int8_t> up_weight(static_cast<size_t>(k) * n);
    std::vector<float> gate_scale(n);
    std::vector<float> up_scale(n);
    for (int inner = 0; inner < k; ++inner) {
      x[inner] = to_bf16(std::sin(static_cast<float>(inner) * 0.017f) * 0.2f);
    }
    for (size_t i = 0; i < gate_weight.size(); ++i) {
      gate_weight[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
      up_weight[i] = static_cast<int8_t>((i * 17 + 11) % 255 - 127);
    }
    for (int col = 0; col < n; ++col) {
      gate_scale[col] = 0.0001f + (col % 31) * 0.000003f;
      up_scale[col] = 0.0001f + (col % 29) * 0.000003f;
    }

    const std::vector<int8_t> gate_kmajor = pack_kmajor(gate_weight, k, n);
    const std::vector<int8_t> up_kmajor = pack_kmajor(up_weight, k, n);
    std::vector<int8_t> gate_blocked = pack_blocked(gate_kmajor, k, n, 32);
    std::vector<int8_t> up_blocked = pack_blocked(up_kmajor, k, n, 32);
    std::vector<uint16_t> direct_output(n);
    std::vector<uint16_t> reference_output(n);
    std::vector<uint16_t> wrapper_output(n);
    std::vector<uint16_t> runtime_output(n);

    SharedObject kernel_library(kernel_path);
    TritonKernel direct =
        kernel_library.symbol<TritonKernel>("_fused_mlp_kernel");
    std::unique_ptr<SharedObject> reference_library;
    TritonKernel reference = nullptr;
    if (argc == 9) {
      reference_library = std::make_unique<SharedObject>(argv[8]);
      reference = reference_library->symbol<TritonKernel>("_fused_mlp_kernel");
    }
    SharedObject runtime_library(runtime_path);
    RuntimeKernel runtime =
        runtime_library.symbol<RuntimeKernel>("fused_mlp_bf16");

    auto run_direct = [&] {
      for (uint32_t pid = 0; pid < static_cast<uint32_t>(grid_x); ++pid) {
        direct(x.data(), gate_blocked.data(), up_blocked.data(),
               gate_scale.data(), up_scale.data(), direct_output.data(), pid, 0,
               0, grid_x, 1, 1);
      }
    };
    auto run_reference = [&] {
      for (uint32_t pid = 0; pid < static_cast<uint32_t>(grid_x); ++pid) {
        reference(x.data(), gate_blocked.data(), up_blocked.data(),
                  gate_scale.data(), up_scale.data(), reference_output.data(),
                  pid, 0, 0, grid_x, 1, 1);
      }
    };

    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> wrapper(
        dirname(kernel_path), "_fused_mlp_kernel");
    void *x_ptr = x.data();
    void *gate_ptr = gate_blocked.data();
    void *up_ptr = up_blocked.data();
    void *gate_scale_ptr = gate_scale.data();
    void *up_scale_ptr = up_scale.data();
    void *output_ptr = wrapper_output.data();
    void *wrapper_args[] = {&x_ptr,          &gate_ptr,     &up_ptr,
                            &gate_scale_ptr, &up_scale_ptr, &output_ptr};
    auto run_wrapper = [&] {
      wrapper.launch_with_signature(grid_x, 1, 1, 1, nullptr, wrapper_args,
                                    "*bf16,*i8,*i8,*fp32,*fp32,*bf16", 6);
    };
    auto run_runtime = [&] {
      runtime(x.data(), gate_kmajor.data(), up_kmajor.data(), gate_scale.data(),
              up_scale.data(), runtime_output.data(), k, n);
    };

    run_direct();
    if (reference)
      run_reference();
    run_wrapper();
    run_runtime();
    if (direct_output != wrapper_output) {
      throw std::runtime_error("libtriton-jit wrapper output mismatch");
    }
    if (reference && direct_output != reference_output) {
      throw std::runtime_error("reference generated output mismatch");
    }
    bool finite_edge_bit_exact = true;
    if (reference) {
      const std::vector<uint16_t> normal_x = x;
      constexpr uint16_t edge_bits[] = {
          0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080, 0x3f80,
          0xbf80, 0x3f81, 0xbf81, 0x4100, 0xc100, 0x4780, 0xc780,
      };
      for (int inner = 0; inner < k; ++inner) {
        x[inner] = edge_bits[inner % std::size(edge_bits)];
      }
      run_direct();
      run_reference();
      finite_edge_bit_exact = direct_output == reference_output;
      if (!finite_edge_bit_exact) {
        throw std::runtime_error("finite BF16 edge-pattern output mismatch");
      }
      x = normal_x;
      run_direct();
      run_reference();
    }
    size_t runtime_mismatch = 0;
    float runtime_max_abs = 0.0f;
    for (int col = 0; col < n; ++col) {
      if (runtime_output[col] != direct_output[col])
        ++runtime_mismatch;
      runtime_max_abs =
          std::max(runtime_max_abs, std::abs(from_bf16(runtime_output[col]) -
                                             from_bf16(direct_output[col])));
    }

    constexpr int warmup = 100;
    const double direct_us = median_us(run_direct, warmup, iterations, batches);
    const double wrapper_us =
        median_us(run_wrapper, warmup, iterations, batches);
    const double runtime_us =
        median_us(run_runtime, warmup, iterations, batches);
    std::cout << "PASS K=" << k << " N=" << n << " BLOCK_N=" << block_n
              << " GRID_X=" << grid_x << '\n'
              << "direct_aot_triton_us=" << direct_us << '\n'
              << "libtriton_jit_wrapper_us=" << wrapper_us << '\n'
              << "legacy_c_runtime_us=" << runtime_us << '\n'
              << "wrapper_minus_direct_us=" << wrapper_us - direct_us << '\n'
              << "wrapper_over_c=" << wrapper_us / runtime_us << "x\n"
              << "legacy_c_mismatch=" << runtime_mismatch << '/' << n << '\n'
              << "legacy_c_max_abs=" << runtime_max_abs << '\n';
    if (reference) {
      const auto [reference_us, active_us] = paired_median_us(
          run_reference, run_direct, warmup, iterations, batches);
      std::cout << "reference_direct_us=" << reference_us << '\n'
                << "active_paired_direct_us=" << active_us << '\n'
                << "active_speedup=" << reference_us / active_us << "x\n"
                << "paired_bit_exact=true\n"
                << "finite_bf16_edges_bit_exact="
                << (finite_edge_bit_exact ? "true" : "false") << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
