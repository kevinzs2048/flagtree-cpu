#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Function>
double median_us(Function &&function, int warmup, int iterations, int batches) {
    for (int i = 0; i < warmup; ++i) {
        function();
    }
    std::vector<double> samples;
    samples.reserve(batches);
    for (int batch = 0; batch < batches; ++batch) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            function();
        }
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count() /
            iterations);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " q4|q8 K [ITERATIONS] [BATCHES]\n";
        return 2;
    }

    try {
        const std::string mode = argv[1];
        const size_t k = std::stoull(argv[2]);
        const int iterations = argc >= 4 ? std::stoi(argv[3]) : 1000;
        const int batches = argc >= 5 ? std::stoi(argv[4]) : 9;
        if ((mode != "q4" && mode != "q8") || k == 0 || k % 32 != 0 ||
            iterations <= 0 || batches <= 0) {
            throw std::runtime_error("invalid mode, K, or repetition count");
        }

        std::vector<float> lhs(k);
        for (size_t index = 0; index < k; ++index) {
            lhs[index] = std::sin(static_cast<float>(index) * 0.017f) * 1.7f;
        }

        size_t packed_size = 0;
        double latency_us = 0.0;
        if (mode == "q4") {
            const size_t mr =
                kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
            const size_t kr =
                kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
            const size_t sr =
                kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
            packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(
                1, k, 32, mr, kr, sr);
            std::vector<uint8_t> packed(packed_size);
            auto run = [&] {
                kai_run_lhs_quant_pack_qsi8d32p_f32(
                    1, k, 32, mr, kr, sr, 0, lhs.data(), k * sizeof(float),
                    packed.data());
            };
            latency_us = median_us(run, 20, iterations, batches);
        } else {
            const size_t mr =
                kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
            const size_t kr =
                kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
            const size_t sr =
                kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
            packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
                1, k, mr, kr, sr);
            std::vector<uint8_t> packed(packed_size);
            auto run = [&] {
                kai_run_lhs_quant_pack_qai8dxp_f32(
                    1, k, mr, kr, sr, 0, lhs.data(), k * sizeof(float),
                    packed.data());
            };
            latency_us = median_us(run, 20, iterations, batches);
        }

        std::cout << "PASS mode=" << mode << " K=" << k << '\n'
                  << "lhs_quant_pack_us=" << latency_us << '\n'
                  << "packed_bytes=" << packed_size << '\n';
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
