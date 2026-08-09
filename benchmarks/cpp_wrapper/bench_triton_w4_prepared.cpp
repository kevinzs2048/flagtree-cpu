#include "triton_w4_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void store_fp16(uint8_t *destination, float value) {
    const __fp16 half = static_cast<__fp16>(value);
    std::memcpy(destination, &half, sizeof(half));
}

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

void check(int status) {
    if (status != 0) {
        throw std::runtime_error(triton_w4_last_error());
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: " << argv[0]
                  << " BUNDLE_DIR K N [ITERATIONS] [BATCHES]\n";
        return 2;
    }

    try {
        const std::string bundle = argv[1];
        const int64_t k = std::stoll(argv[2]);
        const int64_t n = std::stoll(argv[3]);
        const int iterations = argc >= 5 ? std::stoi(argv[4]) : 20;
        const int batches = argc >= 6 ? std::stoi(argv[5]) : 7;
        if (k <= 0 || n <= 0 || k % 32 != 0 || n % 32 != 0 ||
            iterations <= 0 || batches <= 0) {
            throw std::runtime_error("invalid shape or repetition count");
        }

        const int64_t groups = k / 32;
        std::vector<uint8_t> q4(static_cast<size_t>(n) * groups * 18);
        std::vector<uint8_t> q8(static_cast<size_t>(groups) * 34);
        std::vector<float> output(n);

        for (int64_t row = 0; row < n; ++row) {
            for (int64_t group = 0; group < groups; ++group) {
                uint8_t *block =
                    q4.data() + (static_cast<size_t>(row) * groups + group) * 18;
                store_fp16(block, 0.002f + ((row * 7 + group) % 29) * 0.00003f);
                for (int byte = 0; byte < 16; ++byte) {
                    const uint8_t low =
                        static_cast<uint8_t>((row * 3 + group + byte * 5) & 15);
                    const uint8_t high =
                        static_cast<uint8_t>((row + group * 7 + byte * 11) & 15);
                    block[2 + byte] = static_cast<uint8_t>(low | (high << 4));
                }
            }
        }
        for (int64_t group = 0; group < groups; ++group) {
            uint8_t *block = q8.data() + static_cast<size_t>(group) * 34;
            store_fp16(block, 0.001f + (group % 17) * 0.00007f);
            for (int inner = 0; inner < 32; ++inner) {
                block[2 + inner] = static_cast<uint8_t>(
                    static_cast<int8_t>((group * 13 + inner * 17) % 255 - 127));
            }
        }

        check(triton_w4_cache_q4_0(
            bundle.c_str(), q4.data(), q4.data(), k, n));
        check(triton_w4_cached_prepare_q8(q4.data(), q8.data(), k, n));

        auto launch = [&] {
            check(triton_w4_prepared_launch_q8_range(output.data(), 0, n));
        };
        auto prepare = [&] {
            check(triton_w4_cached_prepare_q8(q4.data(), q8.data(), k, n));
        };
        auto prepare_and_launch = [&] {
            prepare();
            check(triton_w4_prepared_launch_q8_range(output.data(), 0, n));
        };

        launch();
        const double launch_us =
            median_us(launch, 10, iterations, batches);
        const double prepare_us =
            median_us(prepare, 10, iterations * 10, batches);
        const double total_us =
            median_us(prepare_and_launch, 10, iterations, batches);
        const double checksum =
            std::accumulate(output.begin(), output.end(), 0.0);

        std::cout << "PASS K=" << k << " N=" << n << '\n'
                  << "triton_static_launch_us=" << launch_us << '\n'
                  << "activation_prepare_us=" << prepare_us << '\n'
                  << "triton_prepare_launch_us=" << total_us << '\n'
                  << "checksum=" << checksum << '\n';
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
