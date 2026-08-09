#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Kernel = void (*)(void *, void *, void *, void *, int32_t, int32_t,
                        int32_t, int32_t, int32_t, int32_t);

struct Library {
  explicit Library(const char *path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle)
      throw std::runtime_error(dlerror());
  }
  ~Library() { dlclose(handle); }
  Kernel kernel() const {
    dlerror();
    void *address = dlsym(handle, "_kai_w8_prefill_kernel");
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<Kernel>(address);
  }
  void *handle;
};

template <typename T> struct AlignedBuffer {
  explicit AlignedBuffer(size_t count) : count(count) {
    void *storage = nullptr;
    if (posix_memalign(&storage, 64, count * sizeof(T)) != 0)
      throw std::bad_alloc();
    data = static_cast<T *>(storage);
  }
  ~AlignedBuffer() { std::free(data); }
  size_t count;
  T *data;
};

template <typename Function>
double timedUs(Function &&function, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration)
    function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}

template <typename First, typename Second>
std::pair<double, double> pairedMedians(First &&first, Second &&second,
                                       int iterations, int batches) {
  for (int warmup = 0; warmup < 100; ++warmup) {
    first();
    second();
  }
  std::vector<double> firstSamples;
  std::vector<double> secondSamples;
  for (int batch = 0; batch < batches; ++batch) {
    if ((batch & 1) == 0) {
      firstSamples.push_back(timedUs(first, iterations));
      secondSamples.push_back(timedUs(second, iterations));
    } else {
      secondSamples.push_back(timedUs(second, iterations));
      firstSamples.push_back(timedUs(first, iterations));
    }
  }
  std::sort(firstSamples.begin(), firstSamples.end());
  std::sort(secondSamples.begin(), secondSamples.end());
  return {firstSamples[firstSamples.size() / 2],
          secondSamples[secondSamples.size() / 2]};
}

void writeFloat(uint8_t *address, float value) {
  std::memcpy(address, &value, sizeof(value));
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 7 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " SVE_SO FIXED_SO M N K ITERATIONS [BATCHES]\n";
    return 2;
  }
  try {
    const size_t m = std::stoull(argv[3]);
    const size_t n = std::stoull(argv[4]);
    const size_t k = std::stoull(argv[5]);
    const int iterations = std::stoi(argv[6]);
    const int batches = argc == 8 ? std::stoi(argv[7]) : 15;
    if (m == 0 || m % 16 != 0 || n == 0 || n % 4 != 0 || k == 0 ||
        k % 32 != 0 || iterations <= 0 || batches <= 0)
      throw std::runtime_error("requires M%16=0, N%4=0 and K%32=0");

    const size_t lhsStride = 4 * k + 32;
    const size_t rhsStride = 4 * k + 48;
    AlignedBuffer<uint8_t> lhs((m / 4) * lhsStride);
    AlignedBuffer<uint8_t> rhs((n / 4) * rhsStride);
    AlignedBuffer<float> sveOutput(m * n);
    AlignedBuffer<float> fixedOutput(m * n);
    std::memset(lhs.data, 0, lhs.count);
    std::memset(rhs.data, 0, rhs.count);

    for (size_t panel = 0; panel < m / 4; ++panel) {
      uint8_t *base = lhs.data + panel * lhsStride;
      for (size_t index = 0; index < 4 * k; ++index)
        base[index] = static_cast<uint8_t>((index * 17 + panel * 13) % 255);
      for (size_t lane = 0; lane < 4; ++lane)
        writeFloat(base + 4 * k + 16 + lane * sizeof(float), 0.001f);
    }
    for (size_t panel = 0; panel < n / 4; ++panel) {
      uint8_t *base = rhs.data + panel * rhsStride;
      for (size_t index = 0; index < 4 * k; ++index)
        base[index] = static_cast<uint8_t>((index * 29 + panel * 7) % 255);
      for (size_t lane = 0; lane < 4; ++lane)
        writeFloat(base + 4 * k + 16 + lane * sizeof(float), 0.001f);
    }

    const float clamp[2] = {-1.0e30f, 1.0e30f};
    Library sveLibrary(argv[1]);
    Library fixedLibrary(argv[2]);
    Kernel sve = sveLibrary.kernel();
    Kernel fixed = fixedLibrary.kernel();
    auto run = [&](Kernel kernel, float *output) {
      for (int32_t pidM = 0; pidM < static_cast<int32_t>(m / 16); ++pidM)
        for (int32_t pidN = 0; pidN < static_cast<int32_t>(n / 4); ++pidN)
          kernel(lhs.data, rhs.data, const_cast<float *>(clamp), output, pidM,
                 pidN, 0, static_cast<int32_t>(m / 16),
                 static_cast<int32_t>(n / 4), 1);
    };
    auto runSve = [&] { run(sve, sveOutput.data); };
    auto runFixed = [&] { run(fixed, fixedOutput.data); };
    runSve();
    runFixed();
    if (std::memcmp(sveOutput.data, fixedOutput.data,
                    m * n * sizeof(float)) != 0)
      throw std::runtime_error("SVE and fixed outputs are not bit-exact");

    const auto [sveUs, fixedUs] =
        pairedMedians(runSve, runFixed, iterations, batches);
    std::cout << std::setprecision(9)
              << "PASS paired Q8 SVE/fixed codegen M=" << m << " N=" << n
              << " K=" << k << '\n'
              << "sve_us=" << sveUs << '\n'
              << "fixed_us=" << fixedUs << '\n'
              << "fixed_over_sve=" << fixedUs / sveUs << "x\n"
              << "bit_exact=true\n"
              << "python_launch=false\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
