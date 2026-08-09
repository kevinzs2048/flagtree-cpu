#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
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

std::vector<int8_t> pack(int k, int n, int blockN) {
  std::vector<int8_t> result(static_cast<size_t>(k) * n);
  for (int outputBlock = 0; outputBlock < n / blockN; ++outputBlock) {
    for (int kBlock = 0; kBlock < k / 4; ++kBlock) {
      for (int outputLane = 0; outputLane < blockN; ++outputLane) {
        for (int kLane = 0; kLane < 4; ++kLane) {
          const int sourceK = 4 * kBlock + kLane;
          const int sourceN = blockN * outputBlock + outputLane;
          const size_t destination =
              (((static_cast<size_t>(outputBlock) * (k / 4) + kBlock) *
                    blockN +
                outputLane) *
                   4 +
               kLane);
          result[destination] = static_cast<int8_t>(
              (static_cast<size_t>(sourceK) * n + sourceN) * 29 + 11);
        }
      }
    }
  }
  return result;
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double measure(Kernel kernel, int blockN, std::vector<int8_t> &x,
               float &xScale, std::vector<int8_t> &packed,
               std::vector<float> &weightScale,
               std::vector<uint16_t> &output, int n, int iterations) {
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    kernel(x.data(), &xScale, packed.data(), weightScale.data(), output.data(),
           0, n / blockN, 0, 0, 0, 1, 1, 1);
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - begin).count() /
         iterations;
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 8 || argc > 10) {
    std::cerr << "usage: " << argv[0]
              << " BASE_SO BASE_BN CANDIDATE_SO CANDIDATE_BN K N ITERS"
                 " [BATCHES] [WARMUP]\n";
    return 2;
  }
  try {
    const int baseBlockN = std::stoi(argv[2]);
    const int candidateBlockN = std::stoi(argv[4]);
    const int k = std::stoi(argv[5]);
    const int n = std::stoi(argv[6]);
    const int iterations = std::stoi(argv[7]);
    const int batches = argc >= 9 ? std::stoi(argv[8]) : 15;
    const int warmup = argc >= 10 ? std::stoi(argv[9]) : 20;
    if (k <= 0 || k % 4 || n <= 0 || baseBlockN <= 0 ||
        candidateBlockN <= 0 || baseBlockN % 4 || candidateBlockN % 4 ||
        n % baseBlockN || n % candidateBlockN || iterations <= 0 ||
        batches <= 0 || warmup < 0)
      throw std::runtime_error("invalid shape, schedule, or sample count");

    Library baseLibrary(argv[1]);
    Library candidateLibrary(argv[3]);
    Kernel base = baseLibrary.kernel();
    Kernel candidate = candidateLibrary.kernel();
    std::vector<int8_t> x(k);
    std::vector<int8_t> basePacked = pack(k, n, baseBlockN);
    std::vector<int8_t> candidatePacked = pack(k, n, candidateBlockN);
    std::vector<float> weightScale(n);
    std::vector<uint16_t> baseOutput(n);
    std::vector<uint16_t> candidateOutput(n);
    float xScale = 0.013f;
    for (int i = 0; i < k; ++i)
      x[i] = static_cast<int8_t>((i * 17 + 5) % 255 - 127);
    for (int i = 0; i < n; ++i)
      weightScale[i] = 0.001f + static_cast<float>(i % 31) * 0.00002f;

    auto call = [&](Kernel kernel, int blockN, std::vector<int8_t> &packed,
                    std::vector<uint16_t> &output) {
      kernel(x.data(), &xScale, packed.data(), weightScale.data(), output.data(),
             0, n / blockN, 0, 0, 0, 1, 1, 1);
    };
    for (int i = 0; i < warmup; ++i) {
      call(base, baseBlockN, basePacked, baseOutput);
      call(candidate, candidateBlockN, candidatePacked, candidateOutput);
    }
    if (baseOutput != candidateOutput)
      throw std::runtime_error("candidate output differs from baseline");

    std::vector<double> baseSamples;
    std::vector<double> candidateSamples;
    for (int batch = 0; batch < batches; ++batch) {
      if ((batch & 1) == 0) {
        baseSamples.push_back(measure(base, baseBlockN, x, xScale, basePacked,
                                      weightScale, baseOutput, n, iterations));
        candidateSamples.push_back(measure(
            candidate, candidateBlockN, x, xScale, candidatePacked,
            weightScale, candidateOutput, n, iterations));
      } else {
        candidateSamples.push_back(measure(
            candidate, candidateBlockN, x, xScale, candidatePacked,
            weightScale, candidateOutput, n, iterations));
        baseSamples.push_back(measure(base, baseBlockN, x, xScale, basePacked,
                                      weightScale, baseOutput, n, iterations));
      }
    }
    const double baseUs = median(std::move(baseSamples));
    const double candidateUs = median(std::move(candidateSamples));
    std::cout << "PASS W8 wide schedule A/B K=" << k << " N=" << n << '\n'
              << "baseline_bn=" << baseBlockN << '\n'
              << "candidate_bn=" << candidateBlockN << '\n'
              << "baseline_us=" << baseUs << '\n'
              << "candidate_us=" << candidateUs << '\n'
              << "speedup=" << baseUs / candidateUs << "x\n"
              << "bit_exact=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
