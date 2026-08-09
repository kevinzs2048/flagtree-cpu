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

using PackKernel = void (*)(void *, void *, void *, int32_t, int32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

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
  PackKernel symbol(const char *name) const {
    dlerror();
    void *address = dlsym(handle, name);
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<PackKernel>(address);
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
  if (argc < 6 || argc > 10) {
    std::cerr << "usage: " << argv[0]
              << " LEGACY_SO ROUNDEVEN_SO M K ITERS [BATCHES] [WARMUP] "
                 "[row|decode|panel4] [SECOND_SYMBOL]\n";
    return 2;
  }
  try {
    const int m = std::stoi(argv[3]);
    const int k = std::stoi(argv[4]);
    const int iterations = std::stoi(argv[5]);
    const int batches = argc >= 7 ? std::stoi(argv[6]) : 21;
    const int warmup = argc >= 8 ? std::stoi(argv[7]) : 200;
    const std::string mode = argc >= 9 ? argv[8] : "row";
    if (m <= 0 || k <= 0 || k % 32 || iterations <= 0 || batches <= 0 ||
        warmup < 0 || (mode == "row" && m % 4) || (mode == "decode" && m > 3) ||
        (mode == "panel4" && m % 4) ||
        (mode != "row" && mode != "decode" && mode != "panel4"))
      throw std::runtime_error(
          "requires K%32=0, row/panel4 M%4=0 or decode M<=3, and "
          "positive counts");

    SharedObject legacyLibrary(argv[1]);
    SharedObject roundevenLibrary(argv[2]);
    const char *firstSymbol = mode == "decode"
                                  ? "_pack_lhs_qsi8d32p_decode_kernel"
                                  : "_pack_lhs_qsi8d32p_row_kernel";
    const char *secondSymbol = argc >= 10 ? argv[9]
                               : mode == "panel4"
                                   ? "_pack_lhs_qsi8d32p_panel4_scalar_kernel"
                                   : firstSymbol;
    PackKernel legacy = legacyLibrary.symbol(firstSymbol);
    PackKernel roundeven = roundevenLibrary.symbol(secondSymbol);

    std::vector<uint16_t> input(static_cast<size_t>(m) * k);
    const size_t packedBytes = mode != "decode"
                                   ? static_cast<size_t>(m / 4) * (k / 32) * 136
                                   : static_cast<size_t>(m) * (k / 32) * 34;
    std::vector<uint8_t> legacyPacked(packedBytes);
    std::vector<uint8_t> roundevenPacked(packedBytes);
    for (size_t i = 0; i < input.size(); ++i) {
      const float value = std::sin(static_cast<float>(i) * 0.017f) * 1.75f +
                          std::cos(static_cast<float>(i) * 0.031f) * 0.125f;
      input[i] = floatToBf16(value);
    }

    auto run = [&](PackKernel kernel, std::vector<uint8_t> &packed,
                   bool panel4) {
      const int programs = panel4 ? m / 4 : m;
      for (int pid = 0; pid < programs; ++pid)
        kernel(input.data(), packed.data(), packed.data(), m, k, pid, 0, 0,
               programs, 1, 1);
    };
    auto runLegacy = [&] { run(legacy, legacyPacked, false); };
    auto runRoundeven = [&] {
      run(roundeven, roundevenPacked, mode == "panel4");
    };
    runLegacy();
    runRoundeven();
    if (legacyPacked != roundevenPacked)
      throw std::runtime_error("roundeven packed blob differs from legacy");

    const auto [legacyUs, roundevenUs] =
        pairedMedianUs(runLegacy, runRoundeven, warmup, iterations, batches);
    std::cout << "PASS Q4 LHS direct-call mode=" << mode << " M=" << m
              << " K=" << k << '\n'
              << (mode == "panel4" ? "row_programs_us="
                                   : "legacy_manual_rne_us=")
              << legacyUs << '\n'
              << (mode == "panel4" ? "panel4_scalar_us=" : "llvm_roundeven_us=")
              << roundevenUs << '\n'
              << "speedup=" << legacyUs / roundevenUs << "x\n"
              << "bit_exact_blob=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
