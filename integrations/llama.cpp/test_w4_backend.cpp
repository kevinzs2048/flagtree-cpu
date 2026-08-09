#include "triton_w4_backend.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
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
        std::cerr << "usage: " << argv[0] << " KERNEL_SO K N\n";
        return 2;
    }
    const char *kernel_path = argv[1];
    const int k = std::stoi(argv[2]);
    const int n = std::stoi(argv[3]);
    if (k % 32 || n % 32) {
        throw std::runtime_error("K/N must be divisible by 32");
    }
    const int groups = k / 32;
    std::vector<uint8_t> q4(
        static_cast<size_t>(n) * groups * 18);
    std::vector<uint8_t> q8(static_cast<size_t>(groups) * 34);
    for (int row = 0; row < n; ++row) {
        for (int group = 0; group < groups; ++group) {
            uint8_t *block =
                q4.data() + (static_cast<size_t>(row) * groups + group) * 18;
            store_fp16(
                block, 0.002f + ((row * 7 + group) % 29) * 0.00003f);
            for (int byte = 0; byte < 16; ++byte) {
                const uint8_t low =
                    static_cast<uint8_t>((row * 3 + group + byte * 5) & 15);
                const uint8_t high =
                    static_cast<uint8_t>((row + group * 7 + byte * 11) & 15);
                block[2 + byte] =
                    static_cast<uint8_t>(low | (high << 4));
            }
        }
    }
    for (int group = 0; group < groups; ++group) {
        uint8_t *block = q8.data() + static_cast<size_t>(group) * 34;
        store_fp16(block, 0.001f + (group % 17) * 0.00007f);
        for (int inner = 0; inner < 32; ++inner) {
            block[2 + inner] = static_cast<uint8_t>(
                static_cast<int8_t>((group * 13 + inner * 17) % 255 - 127));
        }
    }

    std::vector<uint8_t> packed(triton_w4_packed_size(k, n));
    std::vector<float> scales(triton_w4_scale_count(k, n));
    std::vector<int8_t> x_q(k);
    std::vector<float> x_scales(groups);
    std::vector<float> output(n);
    std::vector<float> range_output(n);
    std::vector<float> reference(n);
    if (triton_w4_pack_q4_0(
            q4.data(), packed.data(), scales.data(), k, n) != 0 ||
        triton_w4_unpack_q8_0(
            q8.data(), x_q.data(), x_scales.data(), k) != 0) {
        throw std::runtime_error(triton_w4_last_error());
    }

    triton_w4_kernel *kernel =
        triton_w4_kernel_create(kernel_path, k, n);
    if (!kernel) {
        throw std::runtime_error(triton_w4_last_error());
    }
    if (triton_w4_launch_q8(
            kernel, x_q.data(), x_scales.data(), packed.data(),
            scales.data(), output.data()) != 0) {
        throw std::runtime_error(triton_w4_last_error());
    }
    const std::string bundle_dir =
        std::filesystem::path(kernel_path).parent_path().parent_path();
    if (triton_w4_cache_q4_0(
            bundle_dir.c_str(), q4.data(), q4.data(), k, n) != 0 ||
        triton_w4_cached_prepare_q8(
            q4.data(), q8.data(), k, n) != 0 ||
        triton_w4_prepared_launch_q8_range(
            range_output.data(), 0, n / 2) != 0 ||
        triton_w4_prepared_launch_q8_range(
            range_output.data(), n / 2, n) != 0) {
        throw std::runtime_error(triton_w4_last_error());
    }

    for (int row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (int group = 0; group < groups; ++group) {
            const uint8_t *wb =
                q4.data() + (static_cast<size_t>(row) * groups + group) * 18;
            const uint8_t *xb =
                q8.data() + static_cast<size_t>(group) * 34;
            int32_t dot = 0;
            for (int byte = 0; byte < 16; ++byte) {
                dot +=
                    (static_cast<int>(wb[2 + byte] & 15) - 8) *
                    static_cast<int8_t>(xb[2 + byte]);
                dot +=
                    (static_cast<int>(wb[2 + byte] >> 4) - 8) *
                    static_cast<int8_t>(xb[2 + 16 + byte]);
            }
            sum += static_cast<float>(dot) *
                   load_fp16(wb) * load_fp16(xb);
        }
        reference[row] = sum;
    }

    float max_abs = 0.0f;
    for (int row = 0; row < n; ++row) {
        max_abs = std::max(
            max_abs, std::abs(output[row] - reference[row]));
        max_abs = std::max(
            max_abs, std::abs(range_output[row] - reference[row]));
    }
    triton_w4_kernel_destroy(kernel);
    if (max_abs > 1.2e-5f) {
        std::cerr << "mismatch max_abs=" << max_abs << '\n';
        return 1;
    }
    const size_t released =
        triton_w4_cache_release_range(q4.data(), q4.size());
    if (released != 1 ||
        triton_w4_cached_prepare_q8(
            q4.data(), q8.data(), k, n) == 0) {
        std::cerr << "W4 cache release did not invalidate the weight\n";
        return 1;
    }
    std::cout << "PASS W4 backend K=" << k << " N=" << n
              << " max_abs=" << max_abs
              << " released=" << released
              << " packed_bytes=" << packed.size()
              << " scale_bytes=" << scales.size() * sizeof(float) << '\n';
    return 0;
}
