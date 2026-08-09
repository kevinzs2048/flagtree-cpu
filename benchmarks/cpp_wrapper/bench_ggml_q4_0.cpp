#include "ggml-cpu/quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

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

int main(int argc, char **argv) {
    const int k = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int n = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 100;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 7;
    if (k <= 0 || n <= 0 || k % QK4_0 != 0) {
        std::cerr << "K and N must be positive and K must be divisible by 32\n";
        return 2;
    }

    const int blocks = k / QK4_0;
    std::vector<float> activation(k);
    std::vector<float> weight_fp(static_cast<size_t>(k));
    std::vector<block_q8_0> activation_q(blocks);
    std::vector<block_q4_0> weight_q(static_cast<size_t>(n) * blocks);
    std::vector<float> output(n);

    for (int i = 0; i < k; ++i) {
        activation[i] = std::sin(i * 0.017f) * 1.7f;
    }
    quantize_row_q8_0(activation.data(), activation_q.data(), k);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            weight_fp[col] =
                std::sin((row * 131 + col * 17) * 0.001f) * 0.7f;
        }
        quantize_row_q4_0(
            weight_fp.data(), weight_q.data() + static_cast<size_t>(row) * blocks,
            k);
    }

    auto run = [&] {
        for (int row = 0; row < n; ++row) {
            ggml_vec_dot_q4_0_q8_0(
                k, output.data() + row, 0,
                weight_q.data() + static_cast<size_t>(row) * blocks, 0,
                activation_q.data(), 0, 1);
        }
    };
    run();
    const double latency = median_us(run, 20, iterations, batches);
    const double checksum =
        std::accumulate(output.begin(), output.end(), 0.0);
    std::cout << "PASS K=" << k << " N=" << n << '\n'
              << "ggml_q4_0_q8_0_us=" << latency << '\n'
              << "weight_bytes=" << weight_q.size() * sizeof(block_q4_0) << '\n'
              << "checksum=" << checksum << '\n';
    return 0;
}
