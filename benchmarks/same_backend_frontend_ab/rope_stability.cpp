#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using Kernel = void (*)(void*, void*, void*, void*, uint32_t, uint32_t,
                        uint32_t, uint32_t, uint32_t, uint32_t);

class Library {
 public:
  explicit Library(const char* path) : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle_) throw std::runtime_error(dlerror());
  }
  ~Library() { dlclose(handle_); }
  Kernel symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle_, name);
    if (const char* error = dlerror()) throw std::runtime_error(error);
    return reinterpret_cast<Kernel>(address);
  }

 private:
  void* handle_;
};

class PageBuffer {
 public:
  explicit PageBuffer(size_t bytes) {
    const size_t allocation = (bytes + 8191U) & ~size_t{4095U};
    data_ = static_cast<uint8_t*>(std::aligned_alloc(4096, allocation));
    if (!data_) throw std::bad_alloc();
  }
  ~PageBuffer() { std::free(data_); }
  template <typename Value>
  Value* at(size_t offset = 0) {
    return reinterpret_cast<Value*>(data_ + offset);
  }

 private:
  uint8_t* data_ = nullptr;
};

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

template <typename Function>
double time_us(Function&& function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " ACLE_SO TRITON_SO\n";
    return 2;
  }
  try {
    Library acle_library(argv[1]);
    Library triton_library(argv[2]);
    const Kernel acle = acle_library.symbol("_rope_same_backend_acle");
    const Kernel triton =
        triton_library.symbol("_rope_same_backend_triton");
    constexpr int q_heads = 16;
    constexpr int kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int position = 17;
    const size_t cache_offset = std::getenv("ROPE_CACHE_OFFSET")
                                    ? std::stoul(std::getenv("ROPE_CACHE_OFFSET"))
                                    : 0;
    PageBuffer q_storage(q_heads * head_dim * sizeof(uint16_t));
    PageBuffer k_storage(kv_heads * head_dim * sizeof(uint16_t));
    PageBuffer cache_storage(32 * head_dim * sizeof(uint16_t) + 4096);
    PageBuffer position_storage(4096);
    uint16_t* q = q_storage.at<uint16_t>();
    uint16_t* k = k_storage.at<uint16_t>();
    uint16_t* cache = cache_storage.at<uint16_t>(cache_offset);
    int64_t* positions = position_storage.at<int64_t>();
    positions[0] = position;
    for (int i = 0; i < q_heads * head_dim; ++i)
      q[i] = to_bf16(std::sin(i * 0.019f));
    for (int i = 0; i < kv_heads * head_dim; ++i)
      k[i] = to_bf16(std::cos(i * 0.023f));
    std::fill(cache, cache + 32 * head_dim, to_bf16(0.0f));
    for (int i = 0; i < head_dim / 2; ++i)
      cache[position * head_dim + i] = to_bf16(1.0f);
    auto call_acle = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
        acle(q, k, positions, cache, pid, 0, 0, q_heads + kv_heads, 1, 1);
    };
    auto call_triton = [&] {
      for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
        triton(q, k, positions, cache, pid, 0, 0, q_heads + kv_heads, 1, 1);
    };
    for (int i = 0; i < 5000; ++i) {
      call_acle();
      call_triton();
    }
    std::vector<double> acle_samples, triton_samples;
    constexpr int iterations = 5000;
    constexpr int batches = 31;
    for (int batch = 0; batch < batches; ++batch) {
      if (batch & 1) {
        triton_samples.push_back(time_us(call_triton, iterations));
        acle_samples.push_back(time_us(call_acle, iterations));
      } else {
        acle_samples.push_back(time_us(call_acle, iterations));
        triton_samples.push_back(time_us(call_triton, iterations));
      }
    }
    std::sort(acle_samples.begin(), acle_samples.end());
    std::sort(triton_samples.begin(), triton_samples.end());
    const double acle_us = acle_samples[batches / 2];
    const double triton_us = triton_samples[batches / 2];
    std::cout << "cache_offset=" << cache_offset << " acle_us=" << acle_us
              << " triton_us=" << triton_us
              << " ratio=" << triton_us / acle_us << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
