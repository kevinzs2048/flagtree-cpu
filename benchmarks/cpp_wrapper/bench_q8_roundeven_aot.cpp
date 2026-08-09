#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using QuantKernel = void (*)(void *, void *, void *, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, uint32_t);

uint16_t floatToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

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
  QuantKernel symbol(const char *name, const char *fallback = nullptr) const {
    dlerror();
    void *address = dlsym(handle, name);
    if (const char *error = dlerror()) {
      if (!fallback)
        throw std::runtime_error(error);
      dlerror();
      address = dlsym(handle, fallback);
      if (const char *fallbackError = dlerror())
        throw std::runtime_error(fallbackError);
    }
    return reinterpret_cast<QuantKernel>(address);
  }
  void *handle = nullptr;
};

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

struct PairedLatency {
  double first;
  double second;
};

template <typename First, typename Second>
PairedLatency pairedMedianUs(First &&first, Second &&second, int warmup,
                             int iterations, int batches) {
  for (int i = 0; i < warmup; ++i) {
    first();
    second();
  }
  std::vector<double> firstSamples;
  std::vector<double> secondSamples;
  firstSamples.reserve(batches);
  secondSamples.reserve(batches);
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
      firstSamples.push_back(measure(first));
      secondSamples.push_back(measure(second));
    } else {
      secondSamples.push_back(measure(second));
      firstSamples.push_back(measure(first));
    }
  }
  return {median(std::move(firstSamples)), median(std::move(secondSamples))};
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 9) {
    std::cerr << "usage: " << argv[0]
              << " FIRST_SO SECOND_SO K ITERS [BATCHES] [WARMUP] "
                 "[FIRST_SYMBOL] [SECOND_SYMBOL]\n";
    return 2;
  }
  try {
    const int k = std::stoi(argv[3]);
    const int iterations = std::stoi(argv[4]);
    const int batches = argc >= 6 ? std::stoi(argv[5]) : 21;
    const int warmup = argc >= 7 ? std::stoi(argv[6]) : 200;
    if (k <= 0 || k % 16 || iterations <= 0 || batches <= 0 || warmup < 0)
      throw std::runtime_error("requires positive K%16=0/ITERS/BATCHES");

    SharedObject legacyLibrary(argv[1]);
    SharedObject roundevenLibrary(argv[2]);
    const char *firstSymbol =
        argc >= 8 ? argv[7] : "_quantize_bf16_w8_rne_legacy_kernel";
    const char *secondSymbol =
        argc >= 9 ? argv[8] : "_quantize_bf16_w8_rne_kernel";
    QuantKernel legacy = legacyLibrary.symbol(
        firstSymbol, argc >= 8 ? nullptr : "_quantize_bf16_w8_kernel");
    QuantKernel roundeven = roundevenLibrary.symbol(
        secondSymbol, argc >= 9 ? nullptr : "_quantize_bf16_w8_kernel");

    std::vector<uint16_t> input(k);
    std::vector<int8_t> legacyQuantized(k);
    std::vector<int8_t> roundevenQuantized(k);
    float legacyScale = 0.0f;
    float roundevenScale = 0.0f;
    for (int i = 0; i < k; ++i) {
      const float value = std::sin(static_cast<float>(i) * 0.017f) * 1.75f +
                          std::cos(static_cast<float>(i) * 0.031f) * 0.125f;
      input[i] = floatToBf16(value);
    }

    auto runLegacy = [&] {
      legacy(input.data(), legacyQuantized.data(), &legacyScale, 0, 0, 0, 1, 1,
             1);
    };
    auto runRoundeven = [&] {
      roundeven(input.data(), roundevenQuantized.data(), &roundevenScale, 0, 0,
                0, 1, 1, 1);
    };
    runLegacy();
    runRoundeven();
    if (legacyQuantized != roundevenQuantized ||
        std::memcmp(&legacyScale, &roundevenScale, sizeof(float)) != 0)
      throw std::runtime_error("roundeven result differs from legacy RNE");

    const auto [legacyUs, roundevenUs] =
        pairedMedianUs(runLegacy, runRoundeven, warmup, iterations, batches);
    std::cout << "PASS Q8 RNE direct-call K=" << k << '\n';
    if (argc >= 8 || argc >= 9) {
      std::cout << "first_direct_us=" << legacyUs << '\n'
                << "second_direct_us=" << roundevenUs << '\n';
    } else {
      std::cout << "legacy_manual_rne_us=" << legacyUs << '\n'
                << "llvm_roundeven_us=" << roundevenUs << '\n';
    }
    std::cout << "speedup=" << legacyUs / roundevenUs << "x\n"
              << "bit_exact=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
