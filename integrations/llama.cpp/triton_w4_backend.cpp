#include "triton_w4_backend.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using DirectKernel = void (*)(void *, void *, void *, void *, void *,
                              int32_t, int32_t,
                              uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t);
using DirectKernelStatic = void (*)(void *, void *, void *, void *, void *,
                                    uint32_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t);
using DirectKernelQ41 = void (*)(void *, void *, void *, void *, void *,
                                 void *, void *,
                                 int32_t, int32_t,
                                 uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t);
using DirectKernelQ41Static = void (*)(
    void *, void *, void *, void *, void *, void *, void *,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
using DirectKernelKai = void (*)(void *, void *, void *, void *,
                                 int32_t, int32_t,
                                 uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t);

thread_local std::string last_error;

int fail(const char *message) {
    last_error = message;
    return -1;
}

int fail(const std::exception &error) {
    last_error = error.what();
    return -1;
}

float fp16_to_fp32(const void *source) {
    __fp16 value;
    std::memcpy(&value, source, sizeof(value));
    return static_cast<float>(value);
}

bool valid_shape(int64_t k, int64_t n) {
    return k > 0 && n > 0 && k % 32 == 0 && n % 32 == 0;
}

size_t kai_packed_size(int64_t k, int64_t n) {
    return static_cast<size_t>(n / 4) * static_cast<size_t>(k / 32) * 72;
}

size_t kai_q41_packed_size(int64_t k, int64_t n) {
    return static_cast<size_t>(n / 4) * static_cast<size_t>(k / 32) * 80;
}

void pack_q4_0_kai(
    const void *q4_blocks_nk, uint8_t *packed, int64_t k, int64_t n) {
    constexpr int64_t block_bytes = 18;
    constexpr int64_t quant_offset = 2;
    const auto *source = static_cast<const uint8_t *>(q4_blocks_nk);
    const int64_t groups = k / 32;
    const int64_t tiles = n / 4;
    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int64_t group = 0; group < groups; ++group) {
            uint8_t *destination =
                packed + (tile * groups + group) * 72;
            for (int64_t lane = 0; lane < 4; ++lane) {
                const int64_t row = tile * 4 + lane;
                const uint8_t *block =
                    source + (row * groups + group) * block_bytes;
                std::memcpy(destination + lane * 2, block, 2);
                for (int64_t segment = 0; segment < 2; ++segment) {
                    uint8_t *quant = destination + 8 +
                        (segment * 4 + lane) * 8;
                    for (int64_t byte = 0; byte < 8; ++byte) {
                        quant[byte] = static_cast<uint8_t>(
                            block[quant_offset + segment * 8 + byte] ^ 0x88);
                    }
                }
            }
        }
    }
}

void pack_q4_1_kai(
    const void *q4_blocks_nk, uint8_t *packed, int64_t k, int64_t n) {
    constexpr int64_t block_bytes = 20;
    constexpr int64_t quant_offset = 4;
    const auto *source = static_cast<const uint8_t *>(q4_blocks_nk);
    const int64_t groups = k / 32;
    const int64_t tiles = n / 4;
    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int64_t group = 0; group < groups; ++group) {
            uint8_t *destination =
                packed + (tile * groups + group) * 80;
            for (int64_t lane = 0; lane < 4; ++lane) {
                const int64_t row = tile * 4 + lane;
                const uint8_t *block =
                    source + (row * groups + group) * block_bytes;
                std::memcpy(destination + lane * 2, block, 2);
                std::memcpy(destination + 8 + lane * 2, block + 2, 2);
                for (int64_t segment = 0; segment < 2; ++segment) {
                    uint8_t *quant = destination + 16 +
                        (segment * 4 + lane) * 8;
                    for (int64_t byte = 0; byte < 8; ++byte) {
                        quant[byte] = static_cast<uint8_t>(
                            block[quant_offset + segment * 8 + byte] ^ 0x88);
                    }
                }
            }
        }
    }
}

struct CachedWeight;
struct CachedWeightQ41;
std::mutex cache_mutex;
std::atomic<uint64_t> cache_generation{1};
std::unordered_map<const void *, std::unique_ptr<CachedWeight>> weight_cache;
std::unordered_map<const void *, std::unique_ptr<CachedWeightQ41>>
    weight_q41_cache;

}  // namespace

struct triton_w4_kernel {
    triton_w4_kernel(const char *path, int64_t k_value, int64_t n_value)
        : k(k_value), n(n_value) {
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            throw std::runtime_error(dlerror());
        }
        dlerror();
        direct = reinterpret_cast<DirectKernel>(
            dlsym(handle, "_w4a8_grouped_gemv_kernel"));
        if (const char *error = dlerror()) {
            dlclose(handle);
            handle = nullptr;
            throw std::runtime_error(error);
        }
        std::string static_path(path);
        const size_t slash = static_path.find_last_of('/');
        static_path.replace(
            slash == std::string::npos ? 0 : slash + 1,
            std::string::npos, "_w4a8_grouped_gemv_static_kernel.so");
        static_handle = dlopen(static_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (static_handle) {
            dlerror();
            static_direct = reinterpret_cast<DirectKernelStatic>(
                dlsym(static_handle, "_w4a8_grouped_gemv_static_kernel"));
            if (dlerror()) {
                dlclose(static_handle);
                static_handle = nullptr;
                static_direct = nullptr;
            }
        } else {
            dlerror();
        }
    }

    ~triton_w4_kernel() {
        if (handle) {
            dlclose(handle);
        }
        if (static_handle) {
            dlclose(static_handle);
        }
    }

    int64_t k;
    int64_t n;
    void *handle = nullptr;
    DirectKernel direct = nullptr;
    void *static_handle = nullptr;
    DirectKernelStatic static_direct = nullptr;
};

namespace {

struct CachedWeight {
    CachedWeight(const char *bundle_dir, const void *source,
                 int64_t k_value, int64_t n_value)
        : k(k_value), n(n_value) {
        const std::string shape_dir =
            std::string(bundle_dir) + "/k" + std::to_string(k) + "-n" +
            std::to_string(n);
        const std::string kai_path =
            shape_dir + "/_kai_w4_layout_split_kernel.so";
        if (!std::getenv("GGML_TRITON_W4_LEGACY_LAYOUT")) {
            kai_handle = dlopen(kai_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        }
        if (kai_handle) {
            dlerror();
            kai_direct = reinterpret_cast<DirectKernelKai>(
                dlsym(kai_handle, "_kai_w4_layout_split_kernel"));
            if (const char *error = dlerror()) {
                dlclose(kai_handle);
                kai_handle = nullptr;
                throw std::runtime_error(error);
            }
            packed.resize(kai_packed_size(k, n));
            pack_q4_0_kai(source, packed.data(), k, n);
            return;
        }
        dlerror();

        packed.resize(triton_w4_packed_size(k, n));
        scales.resize(triton_w4_scale_count(k, n));
        kernel = triton_w4_kernel_create_from_bundle(
            bundle_dir, k_value, n_value);
        if (!kernel) {
            throw std::runtime_error(last_error);
        }
        if (triton_w4_pack_q4_0(
                source, packed.data(), scales.data(), k, n) != 0) {
            triton_w4_kernel_destroy(kernel);
            kernel = nullptr;
            throw std::runtime_error(last_error);
        }
    }

    ~CachedWeight() {
        if (kernel) {
            triton_w4_kernel_destroy(kernel);
        }
        if (kai_handle) {
            dlclose(kai_handle);
        }
    }

    int64_t k;
    int64_t n;
    std::vector<uint8_t> packed;
    std::vector<float> scales;
    triton_w4_kernel *kernel = nullptr;
    void *kai_handle = nullptr;
    DirectKernelKai kai_direct = nullptr;
    float clamp[2] = {
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
};

struct CachedWeightQ41 {
    CachedWeightQ41(const char *bundle_dir, const void *source,
                    int64_t k_value, int64_t n_value)
        : k(k_value), n(n_value) {
        const std::string shape_dir =
            std::string(bundle_dir) + "/k" + std::to_string(k) + "-n" +
            std::to_string(n) + "-q41";
        const std::string kai_path =
            shape_dir + "/_kai_q41_layout_split_kernel.so";
        if (!std::getenv("GGML_TRITON_W4_LEGACY_LAYOUT")) {
            kai_handle = dlopen(kai_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        }
        if (kai_handle) {
            dlerror();
            kai_direct = reinterpret_cast<DirectKernelKai>(
                dlsym(kai_handle, "_kai_q41_layout_split_kernel"));
            if (const char *error = dlerror()) {
                dlclose(kai_handle);
                kai_handle = nullptr;
                throw std::runtime_error(error);
            }
            packed.resize(kai_q41_packed_size(k, n));
            pack_q4_1_kai(source, packed.data(), k, n);
            return;
        }
        dlerror();

        packed.resize(triton_w4_packed_size(k, n));
        scales.resize(triton_w4_scale_count(k, n));
        offsets.resize(triton_w4_scale_count(k, n));
        const std::string path = shape_dir + "/_w4a8_q4_1_gemv_kernel.so";
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            throw std::runtime_error(dlerror());
        }
        dlerror();
        direct = reinterpret_cast<DirectKernelQ41>(
            dlsym(handle, "_w4a8_q4_1_gemv_kernel"));
        if (const char *error = dlerror()) {
            dlclose(handle);
            handle = nullptr;
            throw std::runtime_error(error);
        }
        const std::string static_path =
            shape_dir + "/_w4a8_q4_1_gemv_static_kernel.so";
        static_handle =
            dlopen(static_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (static_handle) {
            dlerror();
            static_direct = reinterpret_cast<DirectKernelQ41Static>(
                dlsym(
                    static_handle,
                    "_w4a8_q4_1_gemv_static_kernel"));
            if (dlerror()) {
                dlclose(static_handle);
                static_handle = nullptr;
                static_direct = nullptr;
            }
        } else {
            dlerror();
        }

        constexpr int64_t block_bytes = 20;
        constexpr int64_t quant_offset = 4;
        const auto *bytes = static_cast<const uint8_t *>(source);
        const int64_t groups = k / 32;
        const int64_t tiles = n / 4;
        for (int64_t tile = 0; tile < tiles; ++tile) {
            for (int64_t group = 0; group < groups; ++group) {
                for (int64_t lane = 0; lane < 4; ++lane) {
                    const int64_t row = tile * 4 + lane;
                    const uint8_t *block =
                        bytes + (row * groups + group) * block_bytes;
                    const size_t scale_index =
                        (tile * groups + group) * 4 + lane;
                    const float d = fp16_to_fp32(block);
                    const float m = fp16_to_fp32(block + 2);
                    scales[scale_index] = d;
                    offsets[scale_index] = m + 8.0f * d;
                    for (int64_t k4 = 0; k4 < 4; ++k4) {
                        for (int64_t k_lane = 0; k_lane < 4; ++k_lane) {
                            const int64_t byte = k4 * 4 + k_lane;
                            packed[
                                (((tile * groups + group) * 4 + k4) * 4 +
                                 lane) * 4 + k_lane] =
                                block[quant_offset + byte];
                        }
                    }
                }
            }
        }
    }

    ~CachedWeightQ41() {
        if (handle) {
            dlclose(handle);
        }
        if (static_handle) {
            dlclose(static_handle);
        }
        if (kai_handle) {
            dlclose(kai_handle);
        }
    }

    int64_t k;
    int64_t n;
    std::vector<uint8_t> packed;
    std::vector<float> scales;
    std::vector<float> offsets;
    void *handle = nullptr;
    DirectKernelQ41 direct = nullptr;
    void *static_handle = nullptr;
    DirectKernelQ41Static static_direct = nullptr;
    void *kai_handle = nullptr;
    DirectKernelKai kai_direct = nullptr;
    float clamp[2] = {
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
};

struct PreparedQ8 {
    CachedWeight *cached = nullptr;
    const void *blocks = nullptr;
    std::vector<int8_t> x_q;
    std::vector<float> x_scales;
};

struct PreparedQ81 {
    CachedWeightQ41 *cached = nullptr;
    const void *blocks = nullptr;
    std::vector<int8_t> x_q;
    std::vector<float> x_scales;
    std::vector<float> x_sums;
};

struct ThreadWeightLookup {
    uint64_t generation = 0;
    std::unordered_map<const void *, CachedWeight *> q40;
    std::unordered_map<const void *, CachedWeightQ41 *> q41;
};

thread_local PreparedQ8 prepared_q8;
thread_local PreparedQ81 prepared_q81;
thread_local ThreadWeightLookup thread_weight_lookup;

void refresh_thread_weight_lookup() {
    const uint64_t generation =
        cache_generation.load(std::memory_order_acquire);
    if (thread_weight_lookup.generation != generation) {
        thread_weight_lookup.q40.clear();
        thread_weight_lookup.q41.clear();
        thread_weight_lookup.generation = generation;
    }
}

}  // namespace

extern "C" triton_w4_kernel *triton_w4_kernel_create(
    const char *kernel_so, int64_t k, int64_t n) {
    try {
        if (!kernel_so || !valid_shape(k, n)) {
            fail("invalid Triton W4 kernel configuration");
            return nullptr;
        }
        return new triton_w4_kernel(kernel_so, k, n);
    } catch (const std::exception &error) {
        fail(error);
        return nullptr;
    }
}

extern "C" triton_w4_kernel *triton_w4_kernel_create_from_bundle(
    const char *bundle_dir, int64_t k, int64_t n) {
    if (!bundle_dir) {
        fail("bundle directory is null");
        return nullptr;
    }
    const std::string path =
        std::string(bundle_dir) + "/k" + std::to_string(k) + "-n" +
        std::to_string(n) + "/_w4a8_grouped_gemv_kernel.so";
    return triton_w4_kernel_create(path.c_str(), k, n);
}

extern "C" void triton_w4_kernel_destroy(triton_w4_kernel *kernel) {
    delete kernel;
}

extern "C" size_t triton_w4_packed_size(int64_t k, int64_t n) {
    if (!valid_shape(k, n)) {
        return 0;
    }
    return static_cast<size_t>(k) * static_cast<size_t>(n) / 2;
}

extern "C" size_t triton_w4_scale_count(int64_t k, int64_t n) {
    if (!valid_shape(k, n)) {
        return 0;
    }
    return static_cast<size_t>(k / 32) * static_cast<size_t>(n);
}

extern "C" int triton_w4_pack_q4_0(
    const void *q4_blocks_nk, uint8_t *packed, float *scales,
    int64_t k, int64_t n) {
    if (!q4_blocks_nk || !packed || !scales || !valid_shape(k, n)) {
        return fail("invalid arguments to triton_w4_pack_q4_0");
    }
    constexpr int64_t block_bytes = 18;
    constexpr int64_t quant_offset = 2;
    const auto *source = static_cast<const uint8_t *>(q4_blocks_nk);
    const int64_t groups = k / 32;
    const int64_t tiles = n / 4;
    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int64_t group = 0; group < groups; ++group) {
            for (int64_t lane = 0; lane < 4; ++lane) {
                const int64_t row = tile * 4 + lane;
                const uint8_t *block =
                    source + (row * groups + group) * block_bytes;
                scales[(tile * groups + group) * 4 + lane] =
                    fp16_to_fp32(block);
                for (int64_t k4 = 0; k4 < 4; ++k4) {
                    for (int64_t k_lane = 0; k_lane < 4; ++k_lane) {
                        const int64_t byte = k4 * 4 + k_lane;
                        packed[
                            (((tile * groups + group) * 4 + k4) * 4 +
                             lane) * 4 + k_lane] =
                            block[quant_offset + byte];
                    }
                }
            }
        }
    }
    return 0;
}

extern "C" int triton_w4_unpack_q8_0(
    const void *q8_blocks, int8_t *x_q, float *x_scales, int64_t k) {
    if (!q8_blocks || !x_q || !x_scales || k <= 0 || k % 32 != 0) {
        return fail("invalid arguments to triton_w4_unpack_q8_0");
    }
    constexpr int64_t block_bytes = 34;
    constexpr int64_t quant_offset = 2;
    const auto *source = static_cast<const uint8_t *>(q8_blocks);
    const int64_t groups = k / 32;
    for (int64_t group = 0; group < groups; ++group) {
        const uint8_t *block = source + group * block_bytes;
        x_scales[group] = fp16_to_fp32(block);
        std::memcpy(
            x_q + group * 32, block + quant_offset, 32);
    }
    return 0;
}

extern "C" int triton_w4_launch_q8(
    triton_w4_kernel *kernel, const int8_t *x_q, const float *x_scales,
    const uint8_t *packed, const float *weight_scales, float *output) {
    if (!kernel || !x_q || !x_scales || !packed || !weight_scales ||
        !output) {
        return fail("invalid arguments to triton_w4_launch_q8");
    }
    kernel->direct(
        const_cast<int8_t *>(x_q), const_cast<float *>(x_scales),
        const_cast<uint8_t *>(packed),
        const_cast<float *>(weight_scales), output,
        0, static_cast<int32_t>(kernel->n / 4),
        0, 0, 0, 1, 1, 1);
    return 0;
}

extern "C" int triton_w4_cached_q4_0_q8_0(
    const char *bundle_dir, const void *q4_blocks_nk,
    const void *q8_blocks, float *output, int64_t k, int64_t n) {
    if (!bundle_dir || !q4_blocks_nk || !q8_blocks || !output ||
        !valid_shape(k, n)) {
        return fail("invalid arguments to triton_w4_cached_q4_0_q8_0");
    }
    if (triton_w4_cache_q4_0(
            bundle_dir, q4_blocks_nk, q4_blocks_nk, k, n) != 0) {
        return -1;
    }
    return triton_w4_cached_launch_q8(
        q4_blocks_nk, q8_blocks, output, k, n);
}

extern "C" int triton_w4_cache_q4_0(
    const char *bundle_dir, const void *weight_key,
    const void *q4_blocks_nk, int64_t k, int64_t n) {
    if (!bundle_dir || !weight_key || !q4_blocks_nk ||
        !valid_shape(k, n)) {
        return fail("invalid arguments to triton_w4_cache_q4_0");
    }
    try {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto found = weight_cache.find(weight_key);
        if (found != weight_cache.end()) {
            if (found->second->k != k || found->second->n != n) {
                return fail("cached W4 weight shape changed");
            }
            return 0;
        }
        auto cached = std::make_unique<CachedWeight>(
            bundle_dir, q4_blocks_nk, k, n);
        if (std::getenv("GGML_TRITON_W4_VERBOSE")) {
            std::fprintf(
                stderr,
                "triton-w4 repack/load key=%p K=%lld N=%lld layout=%s\n",
                weight_key, static_cast<long long>(k),
                static_cast<long long>(n),
                cached->kai_direct ? "kai" : "legacy");
        }
        weight_cache.emplace(weight_key, std::move(cached));
        cache_generation.fetch_add(1, std::memory_order_release);
        return 0;
    } catch (const std::exception &error) {
        return fail(error);
    }
}

extern "C" int triton_w4_cached_launch_q8(
    const void *weight_key, const void *q8_blocks,
    float *output, int64_t k, int64_t n) {
    return triton_w4_cached_launch_q8_range(
        weight_key, q8_blocks, output, k, n, 0, n);
}

extern "C" int triton_w4_cached_launch_q8_range(
    const void *weight_key, const void *q8_blocks,
    float *output, int64_t k, int64_t n,
    int64_t output_begin, int64_t output_end) {
    if (triton_w4_cached_prepare_q8(weight_key, q8_blocks, k, n) != 0) {
        return -1;
    }
    return triton_w4_prepared_launch_q8_range(
        output, output_begin, output_end);
}

extern "C" int triton_w4_cached_prepare_q8(
    const void *weight_key, const void *q8_blocks,
    int64_t k, int64_t n) {
    if (!weight_key || !q8_blocks || !valid_shape(k, n)) {
        return fail("invalid arguments to triton_w4_cached_prepare_q8");
    }
    try {
        refresh_thread_weight_lookup();
        CachedWeight *cached = nullptr;
        auto local = thread_weight_lookup.q40.find(weight_key);
        if (local != thread_weight_lookup.q40.end()) {
            cached = local->second;
        } else {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto found = weight_cache.find(weight_key);
            if (found == weight_cache.end()) {
                return fail("W4 weight was not cached during repack");
            }
            cached = found->second.get();
            if (cached->k != k || cached->n != n) {
                return fail("cached W4 weight shape changed");
            }
            const uint64_t generation =
                cache_generation.load(std::memory_order_relaxed);
            if (thread_weight_lookup.generation != generation) {
                thread_weight_lookup.q40.clear();
                thread_weight_lookup.q41.clear();
                thread_weight_lookup.generation = generation;
            }
            thread_weight_lookup.q40.emplace(weight_key, cached);
        }
        if (cached->k != k || cached->n != n) {
            return fail("cached W4 weight shape changed");
        }
        if (cached->kai_direct) {
            prepared_q8.cached = cached;
            prepared_q8.blocks = q8_blocks;
            return 0;
        }
        prepared_q8.x_q.resize(static_cast<size_t>(k));
        prepared_q8.x_scales.resize(static_cast<size_t>(k / 32));
        if (triton_w4_unpack_q8_0(
                q8_blocks, prepared_q8.x_q.data(),
                prepared_q8.x_scales.data(), k) != 0) {
            prepared_q8.cached = nullptr;
            prepared_q8.blocks = nullptr;
            return -1;
        }
        prepared_q8.cached = cached;
        prepared_q8.blocks = nullptr;
        return 0;
    } catch (const std::exception &error) {
        prepared_q8.cached = nullptr;
        prepared_q8.blocks = nullptr;
        return fail(error);
    }
}

extern "C" int triton_w4_prepared_launch_q8_range(
    float *output, int64_t output_begin, int64_t output_end) {
    CachedWeight *cached = prepared_q8.cached;
    if (!cached || !output) {
        return fail("Q8 activation was not prepared on this thread");
    }
    const int64_t n = cached->n;
    if (output_begin < 0 || output_end < output_begin ||
        output_end > n || output_begin % 4 || output_end % 4) {
        return fail("invalid W4 output range");
    }
    try {
        if (cached->kai_direct) {
            if (!prepared_q8.blocks) {
                return fail("Q8 activation blocks are unavailable");
            }
            cached->kai_direct(
                const_cast<void *>(prepared_q8.blocks),
                cached->packed.data(), cached->clamp, output,
                static_cast<int32_t>(output_begin / 4),
                static_cast<int32_t>(output_end / 4),
                0, 0, 0, 1, 1, 1);
            return 0;
        }
        if (output_begin == 0 && output_end == n &&
            cached->kernel->static_direct) {
            cached->kernel->static_direct(
                prepared_q8.x_q.data(), prepared_q8.x_scales.data(),
                cached->packed.data(), cached->scales.data(), output,
                0, 0, 0, 1, 1, 1);
        } else {
            cached->kernel->direct(
                prepared_q8.x_q.data(), prepared_q8.x_scales.data(),
                cached->packed.data(), cached->scales.data(), output,
                static_cast<int32_t>(output_begin / 4),
                static_cast<int32_t>(output_end / 4),
                0, 0, 0, 1, 1, 1);
        }
        return 0;
    } catch (const std::exception &error) {
        return fail(error);
    }
}

extern "C" int triton_w4_cache_q4_1(
    const char *bundle_dir, const void *weight_key,
    const void *q4_blocks_nk, int64_t k, int64_t n) {
    if (!bundle_dir || !weight_key || !q4_blocks_nk ||
        !valid_shape(k, n)) {
        return fail("invalid arguments to triton_w4_cache_q4_1");
    }
    try {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto found = weight_q41_cache.find(weight_key);
        if (found != weight_q41_cache.end()) {
            if (found->second->k != k || found->second->n != n) {
                return fail("cached Q4_1 weight shape changed");
            }
            return 0;
        }
        if (std::getenv("GGML_TRITON_W4_VERBOSE")) {
            std::fprintf(
                stderr,
                "triton-w4 q4_1 repack/load key=%p K=%lld N=%lld\n",
                weight_key, static_cast<long long>(k),
                static_cast<long long>(n));
        }
        weight_q41_cache.emplace(
            weight_key, std::make_unique<CachedWeightQ41>(
                            bundle_dir, q4_blocks_nk, k, n));
        cache_generation.fetch_add(1, std::memory_order_release);
        return 0;
    } catch (const std::exception &error) {
        return fail(error);
    }
}

extern "C" int triton_w4_cached_launch_q8_1(
    const void *weight_key, const void *q8_blocks,
    float *output, int64_t k, int64_t n) {
    return triton_w4_cached_launch_q8_1_range(
        weight_key, q8_blocks, output, k, n, 0, n);
}

extern "C" int triton_w4_cached_launch_q8_1_range(
    const void *weight_key, const void *q8_blocks,
    float *output, int64_t k, int64_t n,
    int64_t output_begin, int64_t output_end) {
    if (triton_w4_cached_prepare_q8_1(weight_key, q8_blocks, k, n) != 0) {
        return -1;
    }
    return triton_w4_prepared_launch_q8_1_range(
        output, output_begin, output_end);
}

extern "C" int triton_w4_cached_prepare_q8_1(
    const void *weight_key, const void *q8_blocks,
    int64_t k, int64_t n) {
    if (!weight_key || !q8_blocks || !valid_shape(k, n)) {
        return fail("invalid arguments to triton_w4_cached_prepare_q8_1");
    }
    try {
        refresh_thread_weight_lookup();
        CachedWeightQ41 *cached = nullptr;
        auto local = thread_weight_lookup.q41.find(weight_key);
        if (local != thread_weight_lookup.q41.end()) {
            cached = local->second;
        } else {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto found = weight_q41_cache.find(weight_key);
            if (found == weight_q41_cache.end()) {
                return fail("Q4_1 weight was not cached");
            }
            cached = found->second.get();
            if (cached->k != k || cached->n != n) {
                return fail("cached Q4_1 weight shape changed");
            }
            const uint64_t generation =
                cache_generation.load(std::memory_order_relaxed);
            if (thread_weight_lookup.generation != generation) {
                thread_weight_lookup.q40.clear();
                thread_weight_lookup.q41.clear();
                thread_weight_lookup.generation = generation;
            }
            thread_weight_lookup.q41.emplace(weight_key, cached);
        }
        if (cached->k != k || cached->n != n) {
            return fail("cached Q4_1 weight shape changed");
        }

        if (cached->kai_direct) {
            prepared_q81.cached = cached;
            prepared_q81.blocks = q8_blocks;
            return 0;
        }

        constexpr int64_t block_bytes = 36;
        constexpr int64_t quant_offset = 4;
        const auto *source = static_cast<const uint8_t *>(q8_blocks);
        const int64_t groups = k / 32;
        prepared_q81.x_q.resize(static_cast<size_t>(k));
        prepared_q81.x_scales.resize(static_cast<size_t>(groups));
        prepared_q81.x_sums.resize(static_cast<size_t>(groups));
        for (int64_t group = 0; group < groups; ++group) {
            const uint8_t *block = source + group * block_bytes;
            prepared_q81.x_scales[group] = fp16_to_fp32(block);
            prepared_q81.x_sums[group] = fp16_to_fp32(block + 2);
            std::memcpy(
                prepared_q81.x_q.data() + group * 32,
                block + quant_offset, 32);
        }
        prepared_q81.cached = cached;
        prepared_q81.blocks = nullptr;
        return 0;
    } catch (const std::exception &error) {
        prepared_q81.cached = nullptr;
        prepared_q81.blocks = nullptr;
        return fail(error);
    }
}

extern "C" int triton_w4_prepared_launch_q8_1_range(
    float *output, int64_t output_begin, int64_t output_end) {
    CachedWeightQ41 *cached = prepared_q81.cached;
    if (!cached || !output) {
        return fail("Q8_1 activation was not prepared on this thread");
    }
    const int64_t n = cached->n;
    if (output_begin < 0 || output_end < output_begin ||
        output_end > n || output_begin % 4 || output_end % 4) {
        return fail("invalid Q4_1 output range");
    }
    try {
        if (cached->kai_direct) {
            if (!prepared_q81.blocks) {
                return fail("Q8_1 activation blocks are unavailable");
            }
            cached->kai_direct(
                const_cast<void *>(prepared_q81.blocks),
                cached->packed.data(), cached->clamp, output,
                static_cast<int32_t>(output_begin / 4),
                static_cast<int32_t>(output_end / 4),
                0, 0, 0, 1, 1, 1);
        } else if (output_begin == 0 && output_end == n &&
            cached->static_direct) {
            cached->static_direct(
                prepared_q81.x_q.data(), prepared_q81.x_scales.data(),
                prepared_q81.x_sums.data(), cached->packed.data(),
                cached->scales.data(), cached->offsets.data(), output,
                0, 0, 0, 1, 1, 1);
        } else {
            cached->direct(
                prepared_q81.x_q.data(), prepared_q81.x_scales.data(),
                prepared_q81.x_sums.data(),
                cached->packed.data(), cached->scales.data(),
                cached->offsets.data(), output,
                static_cast<int32_t>(output_begin / 4),
                static_cast<int32_t>(output_end / 4),
                0, 0, 0, 1, 1, 1);
        }
        return 0;
    } catch (const std::exception &error) {
        return fail(error);
    }
}

extern "C" size_t triton_w4_cache_release_range(
    const void *base, size_t size) {
    if (!base || size == 0) {
        return 0;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
    auto is_in_range = [begin, size](const void *key) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(key);
        return address >= begin && address - begin < size;
    };

    // These assignments only clear the calling thread. Other inference
    // workers must already be stopped before ggml frees their model buffer,
    // which is also a requirement of the surrounding ggml lifecycle.
    prepared_q8.cached = nullptr;
    prepared_q8.blocks = nullptr;
    prepared_q81.cached = nullptr;
    prepared_q81.blocks = nullptr;
    thread_weight_lookup.q40.clear();
    thread_weight_lookup.q41.clear();
    thread_weight_lookup.generation = 0;

    std::lock_guard<std::mutex> lock(cache_mutex);
    size_t released = 0;
    for (auto it = weight_cache.begin(); it != weight_cache.end();) {
        if (is_in_range(it->first)) {
            it = weight_cache.erase(it);
            ++released;
        } else {
            ++it;
        }
    }
    for (auto it = weight_q41_cache.begin();
         it != weight_q41_cache.end();) {
        if (is_in_range(it->first)) {
            it = weight_q41_cache.erase(it);
            ++released;
        } else {
            ++it;
        }
    }
    if (released != 0) {
        cache_generation.fetch_add(1, std::memory_order_release);
    }
    return released;
}

extern "C" const char *triton_w4_last_error(void) {
    return last_error.c_str();
}
