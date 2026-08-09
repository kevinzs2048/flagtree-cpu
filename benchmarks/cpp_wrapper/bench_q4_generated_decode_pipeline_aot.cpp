#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using PackKernel = void (*)(void *, void *, void *, int32_t, int32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
using MatrixKernel = void (*)(void *, void *, void *, int32_t, int32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t);

struct SharedObject {
  explicit SharedObject(const char *path) {
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(dlerror());
  }
  ~SharedObject() {
    if (handle)
      dlclose(handle);
  }
  template <typename Function> Function symbol(const char *name) const {
    dlerror();
    void *address = dlsym(handle, name);
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }
  void *handle = nullptr;
};

uint16_t floatToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

void storeF16(uint8_t *destination, float value) {
  const __fp16 half = static_cast<__fp16>(value);
  std::memcpy(destination, &half, sizeof(half));
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <typename Function>
double medianUs(Function &&function, int warmup, int iterations, int batches) {
  for (int i = 0; i < warmup; ++i)
    function();
  std::vector<double> samples;
  samples.reserve(batches);
  for (int batch = 0; batch < batches; ++batch) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      function();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count() /
        iterations);
  }
  return median(std::move(samples));
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
              << " PACK_SO MATRIX_SO M N K [ITERS] [BATCHES] [WARMUP] "
                 "[REFERENCE_PACK_SO]\n";
    return 2;
  }
  try {
    const int m = std::stoi(argv[3]);
    const int n = std::stoi(argv[4]);
    const int k = std::stoi(argv[5]);
    const int iterations = argc >= 7 ? std::stoi(argv[6]) : 500;
    const int batches = argc >= 8 ? std::stoi(argv[7]) : 21;
    const int warmup = argc >= 9 ? std::stoi(argv[8]) : 200;
    if (m <= 0 || m > 3 || n <= 0 || n % 4 || k <= 0 || k % 32 ||
        iterations <= 0 || batches <= 0 || warmup < 0)
      throw std::runtime_error(
          "requires M in [1,3], N%4=0, K%32=0 and positive counts");

    SharedObject packLibrary(argv[1]);
    SharedObject matrixLibrary(argv[2]);
    PackKernel pack =
        packLibrary.symbol<PackKernel>("_pack_lhs_qsi8d32p_decode_kernel");
    MatrixKernel matrix =
        matrixLibrary.symbol<MatrixKernel>("_q4_decode_sdot_kai_kernel");
    std::unique_ptr<SharedObject> referenceLibrary;
    PackKernel referencePack = nullptr;
    if (argc >= 10) {
      referenceLibrary = std::make_unique<SharedObject>(argv[9]);
      referencePack = referenceLibrary->symbol<PackKernel>(
          "_pack_lhs_qsi8d32p_decode_kernel");
    }

    const int groups = k / 32;
    const int outputTiles = n / 4;
    std::vector<uint16_t> input(static_cast<size_t>(m) * k);
    std::vector<uint8_t> lhsPacked(static_cast<size_t>(m) * groups * 34);
    std::vector<uint8_t> rhsPacked(static_cast<size_t>(outputTiles) * groups *
                                   72);
    std::vector<uint16_t> output(static_cast<size_t>(m) * n);
    std::vector<uint8_t> referenceLhsPacked(lhsPacked.size());
    std::vector<uint16_t> referenceOutput(output.size());

    for (size_t i = 0; i < input.size(); ++i) {
      const float value = std::sin(static_cast<float>(i) * 0.017f) * 1.75f +
                          std::cos(static_cast<float>(i) * 0.031f) * 0.125f;
      input[i] = floatToBf16(value);
    }
    for (int tile = 0; tile < outputTiles; ++tile) {
      for (int group = 0; group < groups; ++group) {
        uint8_t *block = rhsPacked.data() +
                         (static_cast<size_t>(tile) * groups + group) * 72;
        for (int lane = 0; lane < 4; ++lane)
          storeF16(block + lane * 2,
                   0.001f + ((tile * 7 + group * 3 + lane) % 29) * 0.00011f);
        for (int byte = 0; byte < 64; ++byte) {
          const uint8_t low =
              static_cast<uint8_t>((tile * 13 + group * 5 + byte * 3) & 15);
          const uint8_t high =
              static_cast<uint8_t>((tile * 11 + group * 7 + byte * 5) & 15);
          block[8 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
      }
    }

    auto runPack = [&] {
      for (int row = 0; row < m; ++row)
        pack(input.data(), lhsPacked.data(), lhsPacked.data(), m, k, row, 0, 0,
             m, 1, 1);
    };
    auto runMatrix = [&] {
      for (int row = 0; row < m; ++row)
        matrix(lhsPacked.data(), rhsPacked.data(), output.data(), 0,
               outputTiles, row, 0, 0, m, 1, 1);
    };
    auto runPipeline = [&] {
      runPack();
      runMatrix();
    };
    auto runReferencePack = [&] {
      for (int row = 0; row < m; ++row)
        referencePack(input.data(), referenceLhsPacked.data(),
                      referenceLhsPacked.data(), m, k, row, 0, 0, m, 1, 1);
    };
    auto runReferenceMatrix = [&] {
      for (int row = 0; row < m; ++row)
        matrix(referenceLhsPacked.data(), rhsPacked.data(),
               referenceOutput.data(), 0, outputTiles, row, 0, 0, m, 1, 1);
    };
    auto runReferencePipeline = [&] {
      runReferencePack();
      runReferenceMatrix();
    };
    auto runReferencePipelineTimed = [&] {
      for (int row = 0; row < m; ++row)
        referencePack(input.data(), lhsPacked.data(), lhsPacked.data(), m, k,
                      row, 0, 0, m, 1, 1);
      runMatrix();
    };

    runPipeline();
    const std::vector<uint16_t> expected(output);
    runPipeline();
    if (output != expected)
      throw std::runtime_error(
          "generated decode pipeline is not deterministic");
    if (referencePack != nullptr) {
      runReferencePipeline();
      if (lhsPacked != referenceLhsPacked)
        throw std::runtime_error("active pack differs from reference pack");
      if (output != referenceOutput)
        throw std::runtime_error("active output differs from reference output");
    }

    const double packUs = medianUs(runPack, warmup, iterations, batches);
    const double matrixUs = medianUs(runMatrix, warmup, iterations, batches);
    const double pipelineUs =
        medianUs(runPipeline, warmup, iterations, batches);
    const uint64_t checksum =
        std::accumulate(output.begin(), output.end(), uint64_t{0});
    std::cout << "PASS generated Q4 decode pipeline M=" << m << " N=" << n
              << " K=" << k << '\n'
              << "pack_direct_us=" << packUs << '\n'
              << "matrix_direct_us=" << matrixUs << '\n'
              << "separate_sum_us=" << packUs + matrixUs << '\n'
              << "pipeline_direct_us=" << pipelineUs << '\n'
              << "pipeline_over_separate=" << pipelineUs / (packUs + matrixUs)
              << "x\n"
              << "python_dispatch=false\n"
              << "runtime_compute_call=false\n"
              << "output_checksum=" << checksum << '\n';
    if (referencePack != nullptr) {
      const auto [referenceUs, activeUs] = pairedMedianUs(
          runReferencePipelineTimed, runPipeline, warmup, iterations, batches);
      std::cout << "reference_pipeline_direct_us=" << referenceUs << '\n'
                << "active_pipeline_paired_us=" << activeUs << '\n'
                << "pipeline_speedup=" << referenceUs / activeUs << "x\n"
                << "reference_blob_bit_exact=true\n"
                << "reference_output_bit_exact=true\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
