#include "triton_w8_backend.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using QuantKernel = float (*)(const float *, int8_t *, int64_t);
using RuntimeKernel = void (*)(const int8_t *, float, const int8_t *,
                              const float *, float *, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t);

template <typename Function>
double median_us(Function && function) {
    for (int i = 0; i < 100; ++i) function();
    std::vector<double> samples;
    for (int batch = 0; batch < 9; ++batch) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 2000; ++i) function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count() /
            2000.0);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0]
                  << " KERNEL_DIR TRITON_CPU_RUNTIME_SO\n";
        return 2;
    }
    constexpr int64_t k = 1024;
    constexpr int64_t n = 1024;
    constexpr int64_t block_n = 512;
    std::vector<float> x(k);
    std::vector<int8_t> weight(n * k);
    std::vector<float> scale(n);
    for (int64_t i = 0; i < k; ++i) x[i] = std::sin(i * 0.017f);
    for (int64_t i = 0; i < n * k; ++i) {
        weight[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
    }
    for (int64_t i = 0; i < n; ++i) scale[i] = 0.001f + (i % 31) * 0.00003f;
    std::vector<int8_t> packed(triton_w8_packed_size(k, n));
    std::vector<int8_t> xq(k);
    std::vector<float> output(n);
    std::vector<float> reference(n);
    if (triton_w8_pack_i8_nk(
            weight.data(), packed.data(), k, n, block_n) != 0) {
        throw std::runtime_error(triton_w8_last_error());
    }
    triton_w8_kernel * kernel =
        triton_w8_kernel_create(argv[1], k, n, block_n);
    if (!kernel) throw std::runtime_error(triton_w8_last_error());
    if (triton_w8_launch_f32(
            kernel, x.data(), packed.data(), scale.data(), output.data()) != 0) {
        throw std::runtime_error(triton_w8_last_error());
    }

    void * runtime = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!runtime) throw std::runtime_error(dlerror());
    auto quant = reinterpret_cast<QuantKernel>(
        dlsym(runtime, "sdot_quant_act_f32"));
    auto gemv = reinterpret_cast<RuntimeKernel>(
        dlsym(runtime, "sdot_gemv_blk_prequant_f32_range"));
    const float x_scale = quant(x.data(), xq.data(), k);
    gemv(xq.data(), x_scale, packed.data(), scale.data(), reference.data(),
         k, n, n / 4, block_n / 4, 0, n / block_n);
    if (output != reference) {
        throw std::runtime_error("Triton operator backend output mismatch");
    }
    auto run_backend = [&] {
        if (triton_w8_launch_f32(
                kernel, x.data(), packed.data(), scale.data(),
                output.data()) != 0) {
            throw std::runtime_error(triton_w8_last_error());
        }
    };
    auto run_runtime = [&] {
        const float xs = quant(x.data(), xq.data(), k);
        gemv(xq.data(), xs, packed.data(), scale.data(), reference.data(),
             k, n, n / 4, block_n / 4, 0, n / block_n);
    };
    const double backend_us = median_us(run_backend);
    const double runtime_us = median_us(run_runtime);

    std::fill(output.begin(), output.end(), 0.0f);
    if (triton_w8_launch_f32_range(
            kernel, x.data(), packed.data(), scale.data(), output.data(),
            0, 1) != 0 ||
        triton_w8_launch_f32_range(
            kernel, x.data(), packed.data(), scale.data(), output.data(),
            1, 1) != 0 ||
        output != reference) {
        throw std::runtime_error("range launch output mismatch");
    }
    triton_w8_kernel_destroy(kernel);
    dlclose(runtime);
    std::cout << "PASS operator_so_full_and_threadpool_range bit_exact\n"
              << "operator_backend_so_us=" << backend_us << '\n'
              << "ggml_c_runtime_us=" << runtime_us << '\n'
              << "backend_over_c=" << backend_us / runtime_us << "x\n";
    return 0;
}
