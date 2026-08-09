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

using Kernel = void (*)(void*, void*, void*, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t, uint32_t);

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
  PageBuffer() {
    pointer_ = std::aligned_alloc(4096, 8192);
    if (!pointer_) throw std::bad_alloc();
  }
  ~PageBuffer() { std::free(pointer_); }
  uint16_t* at(size_t offset) {
    return reinterpret_cast<uint16_t*>(
        static_cast<uint8_t*>(pointer_) + offset);
  }

 private:
  void* pointer_ = nullptr;
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

struct Timing {
  double acle;
  double triton;
};

template <typename First, typename Second>
Timing paired(First&& first, Second&& second) {
  constexpr int iterations = 20000;
  constexpr int batches = 11;
  for (int i = 0; i < 5000; ++i) {
    first();
    second();
  }
  std::vector<double> a, b;
  for (int batch = 0; batch < batches; ++batch) {
    if (batch & 1) {
      b.push_back(time_us(second, iterations));
      a.push_back(time_us(first, iterations));
    } else {
      a.push_back(time_us(first, iterations));
      b.push_back(time_us(second, iterations));
    }
  }
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return {a[a.size() / 2], b[b.size() / 2]};
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
    const Kernel acle = acle_library.symbol("_rms_same_backend_acle");
    const Kernel triton =
        triton_library.symbol("_rms_same_backend_triton");
    PageBuffer input_storage, weight_storage, output_storage;
    constexpr int n = 1024;
    std::vector<size_t> output_offsets;
    if (std::getenv("SAME_BACKEND_ALIGNMENT_SHORT")) {
      output_offsets = {0, 16, 32, 48, 64};
    } else {
      output_offsets = {16, 32};
      for (size_t offset = 0; offset < 4096; offset += 64)
        output_offsets.push_back(offset);
    }
    std::vector<double> ratios;
    for (size_t output_offset : output_offsets) {
      uint16_t* input = input_storage.at(0);
      uint16_t* weight = weight_storage.at(0);
      uint16_t* output = output_storage.at(output_offset);
      for (int i = 0; i < n; ++i) {
        input[i] = to_bf16(std::sin(i * 0.017f));
        weight[i] = to_bf16(std::cos(i * 0.013f));
      }
      auto call_acle = [&] { acle(input, weight, output, 0, 0, 0, 1, 1, 1); };
      auto call_triton = [&] {
        triton(input, weight, output, 0, 0, 0, 1, 1, 1);
      };
      const Timing timing = paired(call_acle, call_triton);
      const double ratio = timing.triton / timing.acle;
      ratios.push_back(ratio);
      std::cout << "input_offset=0 weight_offset=0 output_offset="
                << output_offset << " acle_us=" << timing.acle
                << " triton_us=" << timing.triton << " ratio=" << ratio
                << '\n';
    }
    std::sort(ratios.begin(), ratios.end());
    std::cout << "ratio_min=" << ratios.front()
              << " ratio_median=" << ratios[ratios.size() / 2]
              << " ratio_max=" << ratios.back() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
