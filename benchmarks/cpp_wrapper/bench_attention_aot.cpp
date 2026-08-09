#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {
using TritonKernel = void (*)(void*, void*, void*, void*, int32_t, float,
                              int32_t, int32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t);
using RuntimeKernel = void (*)(const uint16_t*, const uint16_t*,
                              const uint16_t*, uint16_t*, int64_t, int64_t,
                              float, int64_t, int64_t, int64_t, int64_t);

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

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float from_bf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

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
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    std::cerr << "usage: " << argv[0]
              << " KERNEL_SO RUNTIME_SO SEQ_LEN ITERS [BATCHES]\n";
    return 2;
  }
  try {
    const std::string kernel_path = argv[1];
    const std::string runtime_path = argv[2];
    const int seq_len = std::stoi(argv[3]);
    const int iterations = std::stoi(argv[4]);
    const int batches = argc == 6 ? std::stoi(argv[5]) : 9;
    constexpr int hq = 16;
    constexpr int hkv = 8;
    constexpr int head_dim = 128;
    constexpr float scale = 0.08838834764831845f;

    std::vector<uint16_t> q(hq * head_dim);
    std::vector<uint16_t> k(
        static_cast<size_t>(hkv) * seq_len * head_dim);
    std::vector<uint16_t> v(k.size());
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = to_bf16(std::sin(i * 0.017f));
    }
    for (size_t i = 0; i < k.size(); ++i) {
      k[i] = to_bf16(std::sin(i * 0.013f));
      v[i] = to_bf16(std::cos(i * 0.019f));
    }
    std::vector<uint16_t> direct_output(hq * head_dim);
    std::vector<uint16_t> wrapper_output(hq * head_dim);
    std::vector<uint16_t> runtime_output(hq * head_dim);

    SharedObject kernel_library(kernel_path);
    TritonKernel direct = kernel_library.symbol<TritonKernel>(
        "_flash_attn_decode_codegen_kernel");
    SharedObject runtime_library(runtime_path);
    RuntimeKernel runtime =
        runtime_library.symbol<RuntimeKernel>("flash_attn_decode_bf16");

    auto run_direct = [&] {
      for (uint32_t pid = 0; pid < hq; ++pid) {
        direct(q.data(), k.data(), v.data(), direct_output.data(), seq_len,
               scale, hq, hkv, pid, 0, 0, hq, 1, 1);
      }
    };
    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> wrapper(
        dirname(kernel_path), "_flash_attn_decode_codegen_kernel");
    void* q_ptr = q.data();
    void* k_ptr = k.data();
    void* v_ptr = v.data();
    void* out_ptr = wrapper_output.data();
    int32_t seq_arg = seq_len;
    float scale_arg = scale;
    int32_t hq_arg = hq;
    int32_t hkv_arg = hkv;
    void* args[] = {&q_ptr, &k_ptr, &v_ptr, &out_ptr, &seq_arg,
                    &scale_arg, &hq_arg, &hkv_arg};
    auto run_wrapper = [&] {
      wrapper.launch_with_signature(
          hq, 1, 1, 1, nullptr, args,
          "*bf16,*bf16,*bf16,*bf16,i32,fp32,i32,i32", 8);
    };
    auto run_runtime = [&] {
      runtime(q.data(), k.data(), v.data(), runtime_output.data(), seq_len,
              head_dim, scale, hq, hkv, head_dim, head_dim);
    };
    run_direct();
    run_wrapper();
    run_runtime();
    if (direct_output != wrapper_output) {
      throw std::runtime_error("libtriton-jit wrapper output mismatch");
    }
    size_t runtime_mismatch = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < direct_output.size(); ++i) {
      if (direct_output[i] != runtime_output[i]) ++runtime_mismatch;
      max_abs =
          std::max(max_abs, std::abs(from_bf16(direct_output[i]) -
                                     from_bf16(runtime_output[i])));
    }
    constexpr int warmup = 50;
    const double direct_us =
        median_us(run_direct, warmup, iterations, batches);
    const double wrapper_us =
        median_us(run_wrapper, warmup, iterations, batches);
    const double runtime_us =
        median_us(run_runtime, warmup, iterations, batches);
    std::cout << "PASS N=" << seq_len << " Hq=16 Hkv=8 D=128\n"
              << "direct_aot_triton_us=" << direct_us << '\n'
              << "libtriton_jit_wrapper_us=" << wrapper_us << '\n'
              << "legacy_c_runtime_us=" << runtime_us << '\n'
              << "wrapper_minus_direct_us=" << wrapper_us - direct_us << '\n'
              << "wrapper_over_c=" << wrapper_us / runtime_us << "x\n"
              << "legacy_c_mismatch=" << runtime_mismatch << '/'
              << direct_output.size() << '\n'
              << "legacy_c_max_abs=" << max_abs << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
