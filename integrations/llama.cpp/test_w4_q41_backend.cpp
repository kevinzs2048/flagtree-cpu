#include "triton_w4_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void store_fp16(uint8_t *destination, float value) {
    const __fp16 half = static_cast<__fp16>(value);
    std::memcpy(destination, &half, sizeof(half));
}

float load_fp16(const uint8_t *source) {
    __fp16 half;
    std::memcpy(&half, source, sizeof(half));
    return static_cast<float>(half);
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " BUNDLE_DIR K N\n";
        return 2;
    }
    const int k = std::stoi(argv[2]);
    const int n = std::stoi(argv[3]);
    const int groups = k / 32;
    std::vector<uint8_t> q4(
        static_cast<size_t>(n) * groups * 20);
    std::vector<uint8_t> q8(static_cast<size_t>(groups) * 36);
    for (int row = 0; row < n; ++row) {
        for (int group = 0; group < groups; ++group) {
            uint8_t *block =
                q4.data() + (static_cast<size_t>(row) * groups + group) * 20;
            store_fp16(block, 0.001f + ((row + group) % 19) * 0.00002f);
            store_fp16(block + 2, -0.02f + ((row * 3 + group) % 11) * 0.003f);
            for (int byte = 0; byte < 16; ++byte) {
                const uint8_t low =
                    static_cast<uint8_t>((row + group * 3 + byte * 5) & 15);
                const uint8_t high =
                    static_cast<uint8_t>((row * 7 + group + byte * 9) & 15);
                block[4 + byte] = low | (high << 4);
            }
        }
    }
    for (int group = 0; group < groups; ++group) {
        uint8_t *block = q8.data() + static_cast<size_t>(group) * 36;
        const float d = 0.001f + (group % 13) * 0.00005f;
        int sum = 0;
        for (int inner = 0; inner < 32; ++inner) {
            const int8_t q =
                static_cast<int8_t>((group * 11 + inner * 17) % 255 - 127);
            block[4 + inner] = static_cast<uint8_t>(q);
            sum += q;
        }
        store_fp16(block, d);
        store_fp16(block + 2, d * sum);
    }

    std::vector<float> output(n);
    std::vector<float> range_output(n);
    std::vector<float> reference(n);
    if (triton_w4_cache_q4_1(
            argv[1], q4.data(), q4.data(), k, n) != 0 ||
        triton_w4_cached_launch_q8_1(
            q4.data(), q8.data(), output.data(), k, n) != 0) {
        throw std::runtime_error(triton_w4_last_error());
    }
    if (triton_w4_cached_prepare_q8_1(
            q4.data(), q8.data(), k, n) != 0 ||
        triton_w4_prepared_launch_q8_1_range(
            range_output.data(), 0, n / 2) != 0 ||
        triton_w4_prepared_launch_q8_1_range(
            range_output.data(), n / 2, n) != 0) {
        throw std::runtime_error(triton_w4_last_error());
    }

    for (int row = 0; row < n; ++row) {
        float value = 0.0f;
        for (int group = 0; group < groups; ++group) {
            const uint8_t *wb =
                q4.data() + (static_cast<size_t>(row) * groups + group) * 20;
            const uint8_t *xb =
                q8.data() + static_cast<size_t>(group) * 36;
            int dot = 0;
            for (int byte = 0; byte < 16; ++byte) {
                dot += static_cast<int>(wb[4 + byte] & 15) *
                       static_cast<int8_t>(xb[4 + byte]);
                dot += static_cast<int>(wb[4 + byte] >> 4) *
                       static_cast<int8_t>(xb[20 + byte]);
            }
            value +=
                load_fp16(wb) * load_fp16(xb) * dot +
                load_fp16(wb + 2) * load_fp16(xb + 2);
        }
        reference[row] = value;
    }

    float max_abs = 0.0f;
    for (int row = 0; row < n; ++row) {
        max_abs = std::max(
            max_abs, std::abs(output[row] - reference[row]));
        max_abs = std::max(
            max_abs, std::abs(range_output[row] - reference[row]));
    }
    if (max_abs > 3.0e-5f) {
        std::cerr << "Q4_1 mismatch max_abs=" << max_abs << '\n';
        return 1;
    }
    const size_t released =
        triton_w4_cache_release_range(q4.data(), q4.size());
    if (released != 1 ||
        triton_w4_cached_prepare_q8_1(
            q4.data(), q8.data(), k, n) == 0) {
        std::cerr << "Q4_1 cache release did not invalidate the weight\n";
        return 1;
    }
    std::cout << "PASS Q4_1 backend K=" << k << " N=" << n
              << " max_abs=" << max_abs
              << " released=" << released << '\n';
    return 0;
}
