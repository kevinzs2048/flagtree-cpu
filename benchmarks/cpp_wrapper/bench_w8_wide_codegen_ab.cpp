#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Kernel = void (*)(void *, void *, void *, void *, void *, int32_t,
                        int32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t);

struct Library {
  explicit Library(const char *path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle)
      throw std::runtime_error(dlerror());
  }
  ~Library() { dlclose(handle); }
  Kernel kernel() const {
    dlerror();
    void *symbol = dlsym(handle, "_w8a8_wide_gemv_kernel");
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<Kernel>(symbol);
  }
  void *handle;
};

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double measure(Kernel kernel, std::vector<int8_t> &x, float &xScale,
               std::vector<int8_t> &packed, std::vector<float> &weightScale,
               std::vector<uint16_t> &output, int rangeEnd, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    kernel(x.data(), &xScale, packed.data(), weightScale.data(), output.data(),
           0, rangeEnd, 0, 0, 0, 1, 1, 1);
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 7 || argc > 9) {
    std::cerr << "usage: " << argv[0]
              << " BASE_SO CANDIDATE_SO K N BLOCK_N ITERS [BATCHES] [WARMUP]\n";
    return 2;
  }
  try {
    const int k = std::stoi(argv[3]);
    const int n = std::stoi(argv[4]);
    const int blockN = std::stoi(argv[5]);
    const int iterations = std::stoi(argv[6]);
    const int batches = argc >= 8 ? std::stoi(argv[7]) : 31;
    const int warmup = argc >= 9 ? std::stoi(argv[8]) : 300;
    if (k <= 0 || k % 4 || n <= 0 || n % blockN || blockN % 4 ||
        iterations <= 0 || batches <= 0 || warmup < 0)
      throw std::runtime_error("invalid K/N/BLOCK_N or sample count");

    Library baseLibrary(argv[1]);
    Library candidateLibrary(argv[2]);
    Kernel base = baseLibrary.kernel();
    Kernel candidate = candidateLibrary.kernel();
    std::vector<int8_t> x(k);
    std::vector<int8_t> packed(static_cast<size_t>(k) * n);
    std::vector<float> weightScale(n);
    std::vector<uint16_t> baseOutput(n);
    std::vector<uint16_t> candidateOutput(n);
    float xScale = 0.013f;
    for (int i = 0; i < k; ++i)
      x[i] = static_cast<int8_t>((i * 17 + 5) % 255 - 127);
    for (size_t i = 0; i < packed.size(); ++i)
      packed[i] = static_cast<int8_t>((i * 29 + 11) % 255 - 127);
    for (int i = 0; i < n; ++i)
      weightScale[i] = 0.001f + static_cast<float>(i % 31) * 0.00002f;
    const int rangeEnd = n / blockN;
    auto call = [&](Kernel kernel, std::vector<uint16_t> &output) {
      kernel(x.data(), &xScale, packed.data(), weightScale.data(), output.data(),
             0, rangeEnd, 0, 0, 0, 1, 1, 1);
    };
    for (int i = 0; i < warmup; ++i) {
      call(base, baseOutput);
      call(candidate, candidateOutput);
    }
    if (baseOutput != candidateOutput)
      throw std::runtime_error("candidate output differs from baseline");

    std::vector<double> baseSamples;
    std::vector<double> candidateSamples;
    for (int batch = 0; batch < batches; ++batch) {
      if ((batch & 1) == 0) {
        baseSamples.push_back(measure(base, x, xScale, packed, weightScale,
                                      baseOutput, rangeEnd, iterations));
        candidateSamples.push_back(measure(candidate, x, xScale, packed,
                                            weightScale, candidateOutput,
                                            rangeEnd, iterations));
      } else {
        candidateSamples.push_back(measure(candidate, x, xScale, packed,
                                            weightScale, candidateOutput,
                                            rangeEnd, iterations));
        baseSamples.push_back(measure(base, x, xScale, packed, weightScale,
                                      baseOutput, rangeEnd, iterations));
      }
    }
    const double baseUs = median(std::move(baseSamples));
    const double candidateUs = median(std::move(candidateSamples));
    std::cout << "PASS W8 wide codegen A/B K=" << k << " N=" << n
              << " BN=" << blockN << '\n'
              << "baseline_us=" << baseUs << '\n'
              << "candidate_us=" << candidateUs << '\n'
              << "speedup=" << baseUs / candidateUs << "x\n"
              << "bit_exact=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
