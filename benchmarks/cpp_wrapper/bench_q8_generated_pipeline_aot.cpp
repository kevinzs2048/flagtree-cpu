#include <arm_neon.h>
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

using PackKernel = void (*)(void *, void *, int32_t, int32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
using MatrixKernel = void (*)(void *, void *, void *, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t);

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

float bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void storeF32(uint8_t *destination, float value) {
  std::memcpy(destination, &value, sizeof(value));
}

float32x4_t bf16x4ToFloat(uint16x4_t value) {
  return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(value), 16));
}

void packLhsACLE(const uint16_t *input, uint8_t *packed, int m, int k) {
  const int panelStride = 4 * k + 16;
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t lowerBound = vdupq_n_f32(-128.0f);
  const float32x4_t upperBound = vdupq_n_f32(127.0f);
  for (int row = 0; row < m; ++row) {
    const uint16_t *source = input + static_cast<size_t>(row) * k;
    float32x4_t maximum0 = zero;
    float32x4_t maximum1 = zero;
    for (int offset = 0; offset < k; offset += 8) {
      const uint16x8_t values = vld1q_u16(source + offset);
      maximum0 =
          vmaxq_f32(maximum0, vabsq_f32(bf16x4ToFloat(vget_low_u16(values))));
      maximum1 =
          vmaxq_f32(maximum1, vabsq_f32(bf16x4ToFloat(vget_high_u16(values))));
    }
    const float absmax =
        std::max(vmaxvq_f32(vmaxq_f32(maximum0, maximum1)), 1.0e-8f);
    const float scale = absmax / 127.0f;
    const float32x4_t scaleVector = vdupq_n_f32(scale);
    const int panel = row / 4;
    const int panelRow = row % 4;
    uint8_t *panelBase = packed + static_cast<size_t>(panel) * panelStride;
    for (int offset = 0; offset < k; offset += 8) {
      const uint16x8_t bits = vld1q_u16(source + offset);
      float32x4_t values0 =
          vdivq_f32(bf16x4ToFloat(vget_low_u16(bits)), scaleVector);
      float32x4_t values1 =
          vdivq_f32(bf16x4ToFloat(vget_high_u16(bits)), scaleVector);
      values0 = vmaxq_f32(vminq_f32(values0, upperBound), lowerBound);
      values1 = vmaxq_f32(vminq_f32(values1, upperBound), lowerBound);
      const int16x8_t narrowed =
          vcombine_s16(vqmovn_s32(vcvtq_s32_f32(values0)),
                       vqmovn_s32(vcvtq_s32_f32(values1)));
      vst1_s8(reinterpret_cast<int8_t *>(panelBase) + (offset / 8) * 32 +
                  panelRow * 8,
              vqmovn_s16(narrowed));
    }
    storeF32(panelBase + 4 * k + panelRow * 4, scale);
  }
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

struct PairedTimes {
  double left;
  double right;
};

template <typename Left, typename Right>
PairedTimes pairedMedianUs(Left &&left, Right &&right, int warmup,
                           int iterations, int batches) {
  for (int i = 0; i < warmup; ++i) {
    left();
    right();
  }
  std::vector<double> samples[2];
  samples[0].reserve(batches);
  samples[1].reserve(batches);
  auto measure = [&](auto &&function) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      function();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count() /
           iterations;
  };
  for (int batch = 0; batch < batches; ++batch) {
    if ((batch & 1) == 0) {
      samples[0].push_back(measure(left));
      samples[1].push_back(measure(right));
    } else {
      samples[1].push_back(measure(right));
      samples[0].push_back(measure(left));
    }
  }
  return {median(std::move(samples[0])), median(std::move(samples[1]))};
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 6 || argc > 12) {
    std::cerr << "usage: " << argv[0]
              << " PACK_SO MATRIX_SO M N K [ITERS] [BATCHES] [WARMUP] "
                 "[PACK_SYMBOL] [REFERENCE_PACK_SO] [REFERENCE_PACK_SYMBOL]\n";
    return 2;
  }
  try {
    const int m = std::stoi(argv[3]);
    const int n = std::stoi(argv[4]);
    const int k = std::stoi(argv[5]);
    const int iterations = argc >= 7 ? std::stoi(argv[6]) : 1000;
    const int batches = argc >= 8 ? std::stoi(argv[7]) : 21;
    const int warmup = argc >= 9 ? std::stoi(argv[8]) : 200;
    if (m < 4 || m > 16 || m % 4 || n <= 0 || n % 4 || k <= 0 || k % 32 ||
        iterations <= 0 || batches <= 0 || warmup < 0)
      throw std::runtime_error(
          "requires M in {4,8,12,16}, N%4=0, K%32=0 and positive counts");

    SharedObject packLibrary(argv[1]);
    SharedObject matrixLibrary(argv[2]);
    const char *packSymbol =
        argc >= 10 ? argv[9] : "_pack_lhs_w8_i8mm_kai_kernel";
    PackKernel pack = packLibrary.symbol<PackKernel>(packSymbol);
    std::unique_ptr<SharedObject> referencePackLibrary;
    PackKernel referencePack = nullptr;
    if (argc >= 11) {
      const char *referencePackSymbol =
          argc >= 12 ? argv[11] : "_pack_lhs_w8_i8mm_kai_kernel";
      referencePackLibrary = std::make_unique<SharedObject>(argv[10]);
      referencePack =
          referencePackLibrary->symbol<PackKernel>(referencePackSymbol);
    }
    const char *matrixSymbol = m == 16 ? "_w8_prefill_i8mm_kai_kernel"
                               : m == 12
                                   ? "_w8_prefill_i8mm_kai_m12_kernel"
                                   : "_w8_prefill_i8mm_kai_short_tail_kernel";
    MatrixKernel matrix = matrixLibrary.symbol<MatrixKernel>(matrixSymbol);

    const int lhsPanelStride = 4 * k + 16;
    const int rhsPanelStride = 4 * k + 16;
    const int lhsPanels = m / 4;
    const int rhsPanels = n / 4;
    std::vector<uint16_t> input(static_cast<size_t>(m) * k);
    std::vector<int8_t> logicalWeight(static_cast<size_t>(n) * k);
    std::vector<float> rhsScale(n);
    std::vector<uint8_t> lhsPacked(static_cast<size_t>(lhsPanels) *
                                   lhsPanelStride);
    std::vector<uint8_t> lhsPackedACLE(lhsPacked.size());
    std::vector<uint8_t> rhsPacked(static_cast<size_t>(rhsPanels) *
                                   rhsPanelStride);
    std::vector<uint16_t> output(static_cast<size_t>(m) * n);
    std::vector<uint8_t> referenceLhsPacked(lhsPacked.size());
    std::vector<uint16_t> referenceOutput(output.size());

    for (size_t index = 0; index < input.size(); ++index) {
      const float value = std::sin(static_cast<float>(index) * 0.017f) * 1.7f +
                          std::cos(static_cast<float>(index) * 0.003f) * 0.31f;
      input[index] = floatToBf16(value);
    }
    for (size_t index = 0; index < logicalWeight.size(); ++index)
      logicalWeight[index] =
          static_cast<int8_t>((index * 29 + index / 97 + 7) % 255 - 127);
    for (int column = 0; column < n; ++column)
      rhsScale[column] = 0.001f + (column % 31) * 0.00007f;

    for (int panel = 0; panel < rhsPanels; ++panel) {
      uint8_t *destination =
          rhsPacked.data() + static_cast<size_t>(panel) * rhsPanelStride;
      for (int k8 = 0; k8 < k / 8; ++k8) {
        for (int lane = 0; lane < 4; ++lane) {
          const int column = panel * 4 + lane;
          std::memcpy(destination + k8 * 32 + lane * 8,
                      logicalWeight.data() + static_cast<size_t>(column) * k +
                          k8 * 8,
                      8);
        }
      }
      for (int lane = 0; lane < 4; ++lane)
        storeF32(destination + 4 * k + lane * 4, rhsScale[panel * 4 + lane]);
    }

    auto runPack = [&] {
      for (int pid = 0; pid < m; ++pid)
        pack(input.data(), lhsPacked.data(), m, k, pid, 0, 0, m, 1, 1);
    };
    auto runMatrix = [&] {
      for (int pidN = 0; pidN < rhsPanels; ++pidN)
        matrix(lhsPacked.data(), rhsPacked.data(), output.data(), 0, pidN, 0, 1,
               rhsPanels, 1);
    };
    auto runPipeline = [&] {
      runPack();
      runMatrix();
    };
    auto runReferencePack = [&] {
      for (int pid = 0; pid < m; ++pid)
        referencePack(input.data(), referenceLhsPacked.data(), m, k, pid, 0, 0,
                      m, 1, 1);
    };
    auto runReferenceMatrix = [&] {
      for (int pidN = 0; pidN < rhsPanels; ++pidN)
        matrix(referenceLhsPacked.data(), rhsPacked.data(),
               referenceOutput.data(), 0, pidN, 0, 1, rhsPanels, 1);
    };
    auto runReferencePipeline = [&] {
      runReferencePack();
      runReferenceMatrix();
    };
    auto runReferencePackTimed = [&] {
      for (int pid = 0; pid < m; ++pid)
        referencePack(input.data(), lhsPacked.data(), m, k, pid, 0, 0, m, 1, 1);
    };
    auto runReferencePipelineTimed = [&] {
      runReferencePackTimed();
      runMatrix();
    };

    runPipeline();
    packLhsACLE(input.data(), lhsPackedACLE.data(), m, k);
    if (lhsPacked != lhsPackedACLE)
      throw std::runtime_error("generated and ACLE LHS packs differ");
    if (referencePack != nullptr) {
      runReferencePipeline();
      if (lhsPacked != referenceLhsPacked)
        throw std::runtime_error("generated and reference packs differ");
      if (output != referenceOutput)
        throw std::runtime_error("generated and reference outputs differ");
    }
    size_t mismatches = 0;
    int maxBf16Ulp = 0;
    for (int row = 0; row < m; ++row) {
      float absmax = 0.0f;
      for (int inner = 0; inner < k; ++inner)
        absmax =
            std::max(absmax, std::abs(bf16ToFloat(input[row * k + inner])));
      const float lhsScale = std::max(absmax, 1.0e-8f) / 127.0f;
      for (int column = 0; column < n; ++column) {
        int32_t dot = 0;
        for (int inner = 0; inner < k; ++inner) {
          const float value = bf16ToFloat(input[row * k + inner]);
          const float scaled = std::clamp(value / lhsScale, -128.0f, 127.0f);
          const int8_t quantized = static_cast<int8_t>(scaled);
          dot += static_cast<int32_t>(quantized) *
                 logicalWeight[static_cast<size_t>(column) * k + inner];
        }
        const uint16_t expected =
            floatToBf16(static_cast<float>(dot) * lhsScale * rhsScale[column]);
        const uint16_t actual = output[static_cast<size_t>(row) * n + column];
        const int distance = std::abs(static_cast<int>(expected) - actual);
        maxBf16Ulp = std::max(maxBf16Ulp, distance);
        mismatches += distance != 0;
      }
    }
    if (mismatches)
      throw std::runtime_error(
          "BF16 mismatch count=" + std::to_string(mismatches) +
          " max_ulp=" + std::to_string(maxBf16Ulp));

    auto runACLEPack = [&] {
      packLhsACLE(input.data(), lhsPackedACLE.data(), m, k);
    };
    const auto [packUs, aclePackUs] =
        pairedMedianUs(runPack, runACLEPack, warmup, iterations, batches);
    const double matrixUs = medianUs(runMatrix, warmup, iterations, batches);
    const double pipelineUs =
        medianUs(runPipeline, warmup, iterations, batches);
    const uint64_t checksum =
        std::accumulate(output.begin(), output.end(), uint64_t{0});
    std::cout << "PASS generated Q8 pipeline M=" << m << " N=" << n
              << " K=" << k << '\n'
              << "pack_direct_us=" << packUs << '\n'
              << "acle_pack_direct_us=" << aclePackUs << '\n'
              << "triton_over_acle_pack=" << packUs / aclePackUs << "x\n"
              << "matrix_direct_us=" << matrixUs << '\n'
              << "separate_sum_us=" << packUs + matrixUs << '\n'
              << "pipeline_direct_us=" << pipelineUs << '\n'
              << "pipeline_over_separate=" << pipelineUs / (packUs + matrixUs)
              << "x\n"
              << "bit_exact=true\n"
              << "python_dispatch=false\n"
              << "runtime_compute_call=false\n"
              << "output_checksum=" << checksum << '\n';
    if (referencePack != nullptr) {
      const auto [referencePackUs, activePackUs] = pairedMedianUs(
          runReferencePackTimed, runPack, warmup, iterations, batches);
      const auto [referencePipelineUs, activePipelineUs] = pairedMedianUs(
          runReferencePipelineTimed, runPipeline, warmup, iterations, batches);
      std::cout << "reference_pack_direct_us=" << referencePackUs << '\n'
                << "active_pack_paired_us=" << activePackUs << '\n'
                << "pack_speedup=" << referencePackUs / activePackUs << "x\n"
                << "reference_pipeline_direct_us=" << referencePipelineUs
                << '\n'
                << "active_pipeline_paired_us=" << activePipelineUs << '\n'
                << "pipeline_speedup=" << referencePipelineUs / activePipelineUs
                << "x\n"
                << "reference_blob_bit_exact=true\n"
                << "reference_output_bit_exact=true\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
