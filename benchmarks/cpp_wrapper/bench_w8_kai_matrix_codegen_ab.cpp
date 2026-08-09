#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using MatrixKernel = void (*)(void *, void *, void *, void *, int32_t, int32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t);

class Library {
public:
  explicit Library(const char *path)
      : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle_)
      throw std::runtime_error(dlerror());
  }
  Library(const Library &) = delete;
  Library &operator=(const Library &) = delete;
  ~Library() { dlclose(handle_); }

  MatrixKernel kernel() const {
    dlerror();
    void *address = dlsym(handle_, "_kai_w8_layout_pointer_kernel");
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<MatrixKernel>(address);
  }

private:
  void *handle_;
};

template <typename Value>
void store(std::vector<uint8_t> &buffer, size_t offset, Value value) {
  if (offset + sizeof(Value) > buffer.size())
    throw std::runtime_error("packed-buffer initialization overflow");
  std::memcpy(buffer.data() + offset, &value, sizeof(Value));
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

template <typename Function>
double timed_us(Function &&function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration)
    function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " BASE_SO CANDIDATE_SO K N [ITERS] [BATCHES] [WARMUP]\n";
    return 2;
  }
  try {
    const int32_t k = std::stoi(argv[3]);
    const int32_t n = std::stoi(argv[4]);
    const int iterations = argc >= 6 ? std::stoi(argv[5]) : 100;
    const int batches = argc >= 7 ? std::stoi(argv[6]) : 31;
    const int warmup = argc >= 8 ? std::stoi(argv[7]) : 100;
    if (k <= 0 || k % 32 || n <= 0 || n % 4 || iterations <= 0 ||
        batches <= 0 || warmup < 0)
      throw std::runtime_error("invalid shape or repetition count");

    Library base_library(argv[1]);
    Library candidate_library(argv[2]);
    MatrixKernel base = base_library.kernel();
    MatrixKernel candidate = candidate_library.kernel();

    constexpr int32_t block_n = 4;
    const size_t rhs_stride = static_cast<size_t>(block_n) * k + 48;
    const int32_t output_blocks = n / block_n;
    std::vector<uint8_t> lhs(static_cast<size_t>(k) + 8);
    std::vector<uint8_t> rhs(static_cast<size_t>(output_blocks) * rhs_stride);
    std::vector<uint16_t> base_output(n);
    std::vector<uint16_t> candidate_output(n);
    const float clamp[2] = {-1.0e30f, 1.0e30f};

    for (int32_t index = 0; index < k; ++index)
      lhs[index] = static_cast<uint8_t>((index * 17 + 5) % 255 - 127);
    store<int32_t>(lhs, k, -3);
    store<float>(lhs, k + 4, 0.013f);
    for (int32_t block = 0; block < output_blocks; ++block) {
      const size_t base_offset = static_cast<size_t>(block) * rhs_stride;
      for (int32_t index = 0; index < block_n * k; ++index)
        rhs[base_offset + index] =
            static_cast<uint8_t>((index * 29 + block * 11 + 7) % 255 - 127);
      for (int lane = 0; lane < block_n; ++lane) {
        store<int32_t>(rhs, base_offset + block_n * k + lane * 4,
                       lane * 13 - 19);
        store<float>(rhs, base_offset + block_n * k + 16 + lane * 4,
                     0.001f + lane * 0.00007f);
        store<float>(rhs, base_offset + block_n * k + 32 + lane * 4,
                     lane * 0.0003f);
      }
    }

    auto call = [&](MatrixKernel kernel, std::vector<uint16_t> &output) {
      kernel(lhs.data(), rhs.data(), const_cast<float *>(clamp), output.data(),
             0, output_blocks, 0, 0, 0, 1, 1, 1);
    };
    for (int iteration = 0; iteration < warmup; ++iteration) {
      call(base, base_output);
      call(candidate, candidate_output);
    }
    if (base_output != candidate_output)
      throw std::runtime_error("candidate output differs from baseline");

    std::vector<double> base_samples;
    std::vector<double> candidate_samples;
    for (int batch = 0; batch < batches; ++batch) {
      auto run_base = [&] { call(base, base_output); };
      auto run_candidate = [&] { call(candidate, candidate_output); };
      if ((batch & 1) == 0) {
        base_samples.push_back(timed_us(run_base, iterations));
        candidate_samples.push_back(timed_us(run_candidate, iterations));
      } else {
        candidate_samples.push_back(timed_us(run_candidate, iterations));
        base_samples.push_back(timed_us(run_base, iterations));
      }
    }
    const double base_us = median(std::move(base_samples));
    const double candidate_us = median(std::move(candidate_samples));
    std::cout << "PASS exact-KAI W8 generated matrix A/B K=" << k
              << " N=" << n << '\n'
              << "baseline_us=" << base_us << '\n'
              << "candidate_us=" << candidate_us << '\n'
              << "candidate_over_baseline=" << candidate_us / base_us
              << "x\n"
              << "bf16_output_bit_exact=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
