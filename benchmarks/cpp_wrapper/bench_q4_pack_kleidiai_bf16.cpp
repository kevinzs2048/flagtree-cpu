#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"

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

float bf16ToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

struct SharedObject {
  explicit SharedObject(const char *path) {
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(dlerror());
  }
  ~SharedObject() { dlclose(handle); }
  PackKernel symbol(const char *name) const {
    dlerror();
    void *address = dlsym(handle, name);
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<PackKernel>(address);
  }
  void *handle;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <typename First, typename Second>
std::pair<double, double> pairedMedian(First first, Second second, int warmup,
                                       int iterations, int batches) {
  for (int i = 0; i < warmup; ++i) {
    first();
    second();
  }
  auto measure = [iterations](auto function) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      function();
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - begin)
               .count() /
           iterations;
  };
  std::vector<double> firstSamples;
  std::vector<double> secondSamples;
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
  if (argc < 4 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " TRITON_PANEL4_SO M K [ITERS] [BATCHES] [WARMUP]"
                 " [SYMBOL]\n";
    return 2;
  }
  try {
    const int m = std::stoi(argv[2]);
    const int k = std::stoi(argv[3]);
    const int iterations = argc >= 5 ? std::stoi(argv[4]) : 2000;
    const int batches = argc >= 6 ? std::stoi(argv[5]) : 21;
    const int warmup = argc >= 7 ? std::stoi(argv[6]) : 500;
    const char *symbol =
        argc >= 8 ? argv[7] : "_pack_lhs_qsi8d32p_panel4_scalar_kernel";
    if (m <= 0 || m % 4 || k <= 0 || k % 32 || iterations <= 0 ||
        batches <= 0 || warmup < 0)
      throw std::runtime_error("requires M%4=0, K%32=0, and positive counts");

    constexpr size_t blockLength = 32;
    constexpr size_t mr = 4;
    constexpr size_t kr = 16;
    constexpr size_t sr = 2;
    std::vector<uint16_t> bf16Input(static_cast<size_t>(m) * k);
    std::vector<float> equivalentF32(bf16Input.size());
    for (size_t i = 0; i < bf16Input.size(); ++i) {
      const float source = std::sin(static_cast<float>(i) * 0.017f) * 1.75f +
                           std::cos(static_cast<float>(i) * 0.031f) * 0.125f;
      bf16Input[i] = floatToBf16(source);
      equivalentF32[i] = bf16ToFloat(bf16Input[i]);
    }

    const size_t packedBytes =
        kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(
            m, k, blockLength, mr, kr, sr);
    std::vector<uint8_t> kaiPacked(packedBytes);
    std::vector<uint8_t> tritonPacked(packedBytes);
    SharedObject library(argv[1]);
    PackKernel triton = library.symbol(symbol);

    auto runKai = [&] {
      kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(
          m, k, blockLength, mr, kr, sr, 0, equivalentF32.data(),
          static_cast<size_t>(k) * sizeof(float), kaiPacked.data());
    };
    auto runTriton = [&] {
      for (int pid = 0; pid < m / 4; ++pid)
        triton(bf16Input.data(), tritonPacked.data(), tritonPacked.data(), m, k,
               pid, 0, 0, m / 4, 1, 1);
    };
    auto checkBlob = [&] {
      runKai();
      runTriton();
      if (kaiPacked != tritonPacked) {
        size_t mismatches = 0;
        size_t first = packedBytes;
        for (size_t i = 0; i < packedBytes; ++i) {
          if (kaiPacked[i] != tritonPacked[i]) {
            ++mismatches;
            first = std::min(first, i);
          }
        }
        throw std::runtime_error(
            "packed blob mismatch count=" + std::to_string(mismatches) +
            " first=" + std::to_string(first));
      }
    };
    checkBlob();

    constexpr uint16_t edgeBits[] = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80,
        0xbf80, 0x3f00, 0xbf00, 0x4000, 0xc000, 0x7f7f, 0xff7f, 0x3f81, 0xbf81,
    };
    for (size_t i = 0; i < bf16Input.size(); ++i) {
      bf16Input[i] = edgeBits[i % std::size(edgeBits)];
      equivalentF32[i] = bf16ToFloat(bf16Input[i]);
    }
    checkBlob();

    for (size_t i = 0; i < bf16Input.size(); ++i) {
      const float source = std::sin(static_cast<float>(i) * 0.017f) * 1.75f +
                           std::cos(static_cast<float>(i) * 0.031f) * 0.125f;
      bf16Input[i] = floatToBf16(source);
      equivalentF32[i] = bf16ToFloat(bf16Input[i]);
    }
    checkBlob();

    const auto [kaiUs, tritonUs] =
        pairedMedian(runKai, runTriton, warmup, iterations, batches);
    std::cout << "PASS Q4 BF16-equivalent LHS pack M=" << m << " K=" << k
              << '\n'
              << "kleidiai_f32_equivalent_us=" << kaiUs << '\n'
              << "triton_bf16_panel4_us=" << tritonUs << '\n'
              << "triton_over_kleidiai=" << tritonUs / kaiUs << "x\n"
              << "bit_exact_kai_blob=true\n"
              << "bit_exact_finite_bf16_edges=true\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
