#include <dlfcn.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double measured_us_per_iteration = 0.0;

using ThreePointerKernel = void (*)(void*, void*, void*, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t, uint32_t);
using FourPointerKernel = void (*)(void*, void*, void*, void*, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t);
using W8Kernel = void (*)(void*, void*, void*, void*, int32_t, int32_t,
                          uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                          uint32_t);

class Library {
 public:
  explicit Library(const char* path) : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle_) throw std::runtime_error(dlerror());
  }
  ~Library() { dlclose(handle_); }
  template <typename Function>
  Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle_, name);
    if (const char* error = dlerror()) throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }

 private:
  void* handle_;
};

template <typename Value>
class AlignedArray {
 public:
  explicit AlignedArray(size_t size) : size_(size) {
    const size_t bytes = size * sizeof(Value);
    const size_t allocation = (bytes + 4095U) & ~size_t{4095U};
    data_ = static_cast<Value*>(std::aligned_alloc(4096, allocation));
    if (!data_) throw std::bad_alloc();
  }
  ~AlignedArray() { std::free(data_); }
  Value* data() { return data_; }
  const Value* data() const { return data_; }
  size_t size() const { return size_; }
  Value& operator[](size_t index) { return data_[index]; }
  const Value& operator[](size_t index) const { return data_[index]; }

 private:
  Value* data_ = nullptr;
  size_t size_ = 0;
};

uint16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float from_bf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

template <typename Value>
void store(uint8_t* buffer, size_t offset, Value value) {
  std::memcpy(buffer + offset, &value, sizeof(value));
}

double run_rms(Library& library, const char* symbol, int iterations) {
  const auto kernel = library.symbol<ThreePointerKernel>(symbol);
  constexpr int n = 1024;
  AlignedArray<uint16_t> input(n), weight(n), output(n);
  for (int i = 0; i < n; ++i) {
    input[i] = to_bf16(std::sin(i * 0.017f));
    weight[i] = to_bf16(std::cos(i * 0.013f));
  }
  for (int i = 0; i < 10000; ++i)
    kernel(input.data(), weight.data(), output.data(), 0, 0, 0, 1, 1, 1);
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    kernel(input.data(), weight.data(), output.data(), 0, 0, 0, 1, 1, 1);
  const auto end = std::chrono::steady_clock::now();
  measured_us_per_iteration =
      std::chrono::duration<double, std::micro>(end - begin).count() /
      iterations;
  double checksum = 0.0;
  for (int i = 0; i < n; ++i) checksum += from_bf16(output[i]);
  return checksum;
}

double run_rope(Library& library, const char* symbol, int iterations) {
  const auto kernel = library.symbol<FourPointerKernel>(symbol);
  constexpr int q_heads = 16;
  constexpr int kv_heads = 8;
  constexpr int head_dim = 128;
  constexpr int position = 17;
  constexpr size_t q_size = q_heads * head_dim;
  constexpr size_t k_size = kv_heads * head_dim;
  AlignedArray<uint16_t> q(q_size), k(k_size);
  for (size_t i = 0; i < q.size(); ++i) q[i] = to_bf16(std::sin(i * 0.019f));
  for (size_t i = 0; i < k.size(); ++i) k[i] = to_bf16(std::cos(i * 0.023f));
  AlignedArray<int64_t> positions(1);
  positions[0] = position;
  AlignedArray<uint16_t> cache(32 * head_dim);
  for (size_t i = 0; i < cache.size(); ++i) cache[i] = to_bf16(0.0f);
  for (int i = 0; i < head_dim / 2; ++i)
    cache[position * head_dim + i] = to_bf16(1.0f);
  auto invoke = [&] {
    for (uint32_t pid = 0; pid < q_heads + kv_heads; ++pid)
      kernel(q.data(), k.data(), positions.data(), cache.data(), pid, 0, 0,
             q_heads + kv_heads, 1, 1);
  };
  for (int i = 0; i < 10000; ++i) invoke();
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) invoke();
  const auto end = std::chrono::steady_clock::now();
  measured_us_per_iteration =
      std::chrono::duration<double, std::micro>(end - begin).count() /
      iterations;
  double checksum = 0.0;
  for (size_t i = 0; i < q.size(); ++i) checksum += from_bf16(q[i]);
  for (size_t i = 0; i < k.size(); ++i) checksum += from_bf16(k[i]);
  return checksum;
}

double run_w8(Library& library, const char* symbol, int iterations) {
  const auto kernel = library.symbol<W8Kernel>(symbol);
  constexpr int k = 1024;
  constexpr int n = 3072;
  constexpr int blocks = n / 4;
  constexpr int rhs_stride = 4 * k + 48;
  AlignedArray<uint8_t> lhs(k + 8);
  AlignedArray<uint8_t> rhs(static_cast<size_t>(blocks) * rhs_stride);
  for (int i = 0; i < k; ++i)
    lhs[i] = static_cast<uint8_t>((i * 17 + 5) % 255 - 127);
  store<int32_t>(lhs.data(), k, -3);
  store<float>(lhs.data(), k + 4, 0.013f);
  for (int block = 0; block < blocks; ++block) {
    const size_t base = static_cast<size_t>(block) * rhs_stride;
    for (int i = 0; i < 4 * k; ++i)
      rhs[base + i] =
          static_cast<uint8_t>((i * 29 + block * 11 + 7) % 255 - 127);
    for (int lane = 0; lane < 4; ++lane) {
      store<int32_t>(rhs.data(), base + 4 * k + lane * 4, lane * 13 - 19);
      store<float>(rhs.data(), base + 4 * k + 16 + lane * 4,
                   0.001f + lane * 0.00007f);
      store<float>(rhs.data(), base + 4 * k + 32 + lane * 4,
                   lane * 0.0003f);
    }
  }
  const float clamp[2] = {-1.0e30f, 1.0e30f};
  AlignedArray<float> output(n);
  for (int i = 0; i < 100; ++i)
    kernel(lhs.data(), rhs.data(), const_cast<float*>(clamp), output.data(), 0,
           blocks, 0, 0, 0, 1, 1, 1);
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    kernel(lhs.data(), rhs.data(), const_cast<float*>(clamp), output.data(), 0,
           blocks, 0, 0, 0, 1, 1, 1);
  const auto end = std::chrono::steady_clock::now();
  measured_us_per_iteration =
      std::chrono::duration<double, std::micro>(end - begin).count() /
      iterations;
  double checksum = 0.0;
  for (int i = 0; i < n; ++i) checksum += output[i];
  return checksum;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " rms|rope|w8 SO SYMBOL ITERATIONS\n";
    return 2;
  }
  try {
    const std::string mode = argv[1];
    const int iterations = std::stoi(argv[4]);
    Library library(argv[2]);
    double checksum = 0.0;
    if (mode == "rms")
      checksum = run_rms(library, argv[3], iterations);
    else if (mode == "rope")
      checksum = run_rope(library, argv[3], iterations);
    else if (mode == "w8")
      checksum = run_w8(library, argv[3], iterations);
    else
      throw std::runtime_error("unknown mode");
    std::cout << "mode=" << mode << " iterations=" << iterations
              << " us_per_iteration=" << measured_us_per_iteration
              << " checksum=" << checksum << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
