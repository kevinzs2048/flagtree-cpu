#include "triton_w8_backend.h"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using QuantKernel = float (*)(const float *, int8_t *, int64_t);
using RuntimeKernel = void (*)(const int8_t *, float, const int8_t *,
                              const float *, float *, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t);

template <typename Function>
double median_us(Function && function, int warmup, int iterations, int batches) {
    for (int i = 0; i < warmup; ++i) function();
    std::vector<double> samples;
    for (int batch = 0; batch < batches; ++batch) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count() /
            iterations);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main(int argc, char ** argv) {
    if (argc != 9) {
        std::cerr << "usage: " << argv[0]
                  << " BUNDLE_DIR RUNTIME_SO K N BLOCK_N THREADS ITERS BATCHES\n";
        return 2;
    }
    const char * bundle_dir = argv[1];
    const char * runtime_path = argv[2];
    const int64_t k = std::stoll(argv[3]);
    const int64_t n = std::stoll(argv[4]);
    const int64_t block_n = std::stoll(argv[5]);
    const int threads = std::stoi(argv[6]);
    const int iterations = std::stoi(argv[7]);
    const int batches = std::stoi(argv[8]);
    const int64_t grid_x = n / block_n;
    if (threads <= 0 || grid_x <= 0) {
        throw std::runtime_error("invalid thread/grid configuration");
    }

    std::vector<float> x(k);
    std::vector<int8_t> weight(n * k);
    std::vector<float> scale(n);
    for (int64_t i = 0; i < k; ++i) x[i] = std::sin(i * 0.017f);
    for (int64_t i = 0; i < n * k; ++i) {
        weight[i] = static_cast<int8_t>((i * 29 + 7) % 255 - 127);
    }
    for (int64_t i = 0; i < n; ++i) {
        scale[i] = 0.001f + (i % 31) * 0.00003f;
    }
    std::vector<int8_t> packed(triton_w8_packed_size(k, n));
    std::vector<int8_t> xq(k);
    std::vector<float> split_output(n);
    std::vector<float> full_output(n);
    std::vector<float> runtime_output(n);
    if (triton_w8_pack_i8_nk(
            weight.data(), packed.data(), k, n, block_n) != 0) {
        throw std::runtime_error(triton_w8_last_error());
    }
    triton_w8_split_kernel * kernel =
        triton_w8_split_kernel_create_from_bundle(
            bundle_dir, k, n, block_n);
    if (!kernel) throw std::runtime_error(triton_w8_last_error());

    void * runtime = dlopen(runtime_path, RTLD_NOW | RTLD_LOCAL);
    if (!runtime) throw std::runtime_error(dlerror());
    auto quant = reinterpret_cast<QuantKernel>(
        dlsym(runtime, "sdot_quant_act_f32"));
    auto gemv = reinterpret_cast<RuntimeKernel>(
        dlsym(runtime, "sdot_gemv_blk_prequant_f32_range"));
    if (!quant || !gemv) throw std::runtime_error("missing C runtime symbol");

    if (triton_w8_split_launch_f32(
            kernel, x.data(), packed.data(), scale.data(),
            full_output.data()) != 0) {
        throw std::runtime_error(triton_w8_last_error());
    }
    const float reference_scale = quant(x.data(), xq.data(), k);
    gemv(xq.data(), reference_scale, packed.data(), scale.data(),
         runtime_output.data(), k, n, n / 4, block_n / 4, 0, grid_x);
    if (full_output != runtime_output) {
        throw std::runtime_error("split full-grid output mismatch");
    }
    auto launch_full_split = [&] {
        if (triton_w8_split_launch_f32(
                kernel, x.data(), packed.data(), scale.data(),
                full_output.data()) != 0) {
            throw std::runtime_error(triton_w8_last_error());
        }
    };
    auto launch_full_runtime = [&] {
        const float local_scale = quant(x.data(), xq.data(), k);
        gemv(xq.data(), local_scale, packed.data(), scale.data(),
             runtime_output.data(), k, n, n / 4, block_n / 4, 0, grid_x);
    };
    const double full_split_us =
        median_us(launch_full_split, 50, iterations, batches);
    const double full_runtime_us =
        median_us(launch_full_runtime, 50, iterations, batches);

    enum Mode { split_mode, runtime_mode };
    std::atomic<Mode> mode{split_mode};
    std::atomic<int64_t> next_block{0};
    std::atomic<bool> stop{false};
    float x_scale = 0.0f;
    std::barrier start_barrier(threads + 1);
    std::barrier end_barrier(threads + 1);
    std::barrier quant_barrier(threads);
    std::vector<std::thread> workers;
    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&, tid] {
            while (true) {
                start_barrier.arrive_and_wait();
                if (stop.load(std::memory_order_relaxed)) break;
                const Mode selected = mode.load(std::memory_order_relaxed);
                if (tid == 0) {
                    if (selected == split_mode) {
                        if (triton_w8_split_quantize_f32(
                                kernel, x.data()) != 0) {
                            std::abort();
                        }
                    } else {
                        x_scale = quant(x.data(), xq.data(), k);
                    }
                }
                quant_barrier.arrive_and_wait();
                int64_t block;
                while ((block = next_block.fetch_add(
                            1, std::memory_order_relaxed)) < grid_x) {
                    if (selected == split_mode) {
                        if (triton_w8_split_launch_prequant_f32_range(
                                kernel, packed.data(), scale.data(),
                                split_output.data(), block, 1) != 0) {
                            std::abort();
                        }
                    } else {
                        gemv(xq.data(), x_scale, packed.data(), scale.data(),
                             runtime_output.data(), k, n, n / 4,
                             block_n / 4, block, 1);
                    }
                }
                end_barrier.arrive_and_wait();
            }
        });
    }
    auto launch = [&](Mode selected) {
        mode.store(selected, std::memory_order_relaxed);
        next_block.store(0, std::memory_order_relaxed);
        start_barrier.arrive_and_wait();
        end_barrier.arrive_and_wait();
    };
    launch(split_mode);
    launch(runtime_mode);
    if (split_output != runtime_output) {
        throw std::runtime_error("split threadpool output mismatch");
    }
    const double split_us =
        median_us([&] { launch(split_mode); }, 50, iterations, batches);
    const double runtime_us =
        median_us([&] { launch(runtime_mode); }, 50, iterations, batches);

    stop.store(true, std::memory_order_relaxed);
    start_barrier.arrive_and_wait();
    for (auto & worker : workers) worker.join();
    triton_w8_split_kernel_destroy(kernel);
    dlclose(runtime);
    std::cout << "PASS K=" << k << " N=" << n
              << " BLOCK_N=" << block_n << " THREADS=" << threads << '\n'
              << "split_operator_so_us=" << full_split_us << '\n'
              << "ggml_c_serial_us=" << full_runtime_us << '\n'
              << "serial_triton_over_c="
              << full_split_us / full_runtime_us << "x\n"
              << "split_triton_threadpool_us=" << split_us << '\n'
              << "ggml_c_threadpool_us=" << runtime_us << '\n'
              << "triton_over_c=" << split_us / runtime_us << "x\n";
    return 0;
}
