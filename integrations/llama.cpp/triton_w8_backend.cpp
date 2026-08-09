#include "triton_w8_backend.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"
#include "triton_jit/triton_kernel.h"

namespace {

using DirectKernel = void (*)(void *, void *, void *, void *, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t);
using QuantDirectKernel = void (*)(void *, void *, void *, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t);
using PrequantDirectKernel = void (*)(void *, void *, void *, void *, void *,
                                      uint32_t, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t);

thread_local std::string last_error;

int fail(const char * message) {
    last_error = message;
    return -1;
}

int fail(const std::exception & error) {
    last_error = error.what();
    return -1;
}

}  // namespace

struct triton_w8_kernel {
    triton_w8_kernel(std::string directory, int64_t k_value, int64_t n_value,
                     int64_t block_value)
        : dir(std::move(directory)),
          k(k_value),
          n(n_value),
          block_n(block_value),
          grid_x(n_value / block_value),
          wrapper(dir, "_f32_w8_gemv_kernel") {
        auto handle = triton_jit::CpuBackend::load_kernel(
            dir, "_f32_w8_gemv_kernel");
        direct = reinterpret_cast<DirectKernel>(handle.function);
        if (!direct) {
            throw std::runtime_error("AOT kernel has no callable symbol");
        }
    }

    std::string dir;
    int64_t k;
    int64_t n;
    int64_t block_n;
    int64_t grid_x;
    DirectKernel direct;
    triton_jit::TritonKernelImpl<triton_jit::CpuBackend> wrapper;
};

struct triton_w8_split_kernel {
    triton_w8_split_kernel(
        std::string quant_directory, std::string gemv_directory,
        int64_t k_value, int64_t n_value, int64_t block_value)
        : quant_dir(std::move(quant_directory)),
          gemv_dir(std::move(gemv_directory)),
          k(k_value),
          n(n_value),
          block_n(block_value),
          grid_x(n_value / block_value),
          x_q(static_cast<size_t>(k_value)),
          x_scale(1) {
        auto quant_handle = triton_jit::CpuBackend::load_kernel(
            quant_dir, "_quantize_f32_w8_kernel");
        quant = reinterpret_cast<QuantDirectKernel>(quant_handle.function);
        auto gemv_handle = triton_jit::CpuBackend::load_kernel(
            gemv_dir, "_f32_w8_prequant_gemv_kernel");
        gemv =
            reinterpret_cast<PrequantDirectKernel>(gemv_handle.function);
        if (!quant || !gemv) {
            throw std::runtime_error(
                "split AOT kernel has no callable symbol");
        }
    }

    std::string quant_dir;
    std::string gemv_dir;
    int64_t k;
    int64_t n;
    int64_t block_n;
    int64_t grid_x;
    std::vector<int8_t> x_q;
    std::vector<float> x_scale;
    QuantDirectKernel quant;
    PrequantDirectKernel gemv;
};

bool valid_config(int64_t k, int64_t n, int64_t block_n) {
    return k > 0 && n > 0 && block_n > 0 && k % 4 == 0 &&
           n % block_n == 0 && block_n % 4 == 0;
}

extern "C" triton_w8_kernel * triton_w8_kernel_create(
    const char * kernel_dir, int64_t k, int64_t n, int64_t block_n) {
    try {
        if (!kernel_dir || !valid_config(k, n, block_n)) {
            fail("invalid Triton W8 kernel configuration");
            return nullptr;
        }
        return new triton_w8_kernel(kernel_dir, k, n, block_n);
    } catch (const std::exception & error) {
        fail(error);
        return nullptr;
    }
}

extern "C" triton_w8_kernel * triton_w8_kernel_create_from_bundle(
    const char * bundle_dir, int64_t k, int64_t n, int64_t block_n) {
    if (!bundle_dir) {
        fail("bundle directory is null");
        return nullptr;
    }
    const std::string kernel_dir =
        std::string(bundle_dir) + "/k" + std::to_string(k) + "-n" +
        std::to_string(n) + "-bn" + std::to_string(block_n);
    return triton_w8_kernel_create(
        kernel_dir.c_str(), k, n, block_n);
}

extern "C" void triton_w8_kernel_destroy(triton_w8_kernel * kernel) {
    delete kernel;
}

extern "C" size_t triton_w8_packed_size(int64_t k, int64_t n) {
    if (k <= 0 || n <= 0 || k % 4 != 0 || n % 4 != 0) {
        return 0;
    }
    return static_cast<size_t>(k) * static_cast<size_t>(n);
}

extern "C" int triton_w8_pack_i8_nk(
    const int8_t * weight_nk, int8_t * packed, int64_t k, int64_t n,
    int64_t block_n) {
    if (!weight_nk || !packed || k <= 0 || n <= 0 || block_n <= 0 ||
        k % 4 != 0 || n % block_n != 0 || block_n % 4 != 0) {
        return fail("invalid arguments to triton_w8_pack_i8_nk");
    }
    const int64_t k4 = k / 4;
    const int64_t groups = block_n / 4;
    const int64_t blocks = n / block_n;
    for (int64_t block = 0; block < blocks; ++block) {
        for (int64_t kb = 0; kb < k4; ++kb) {
            for (int64_t group = 0; group < groups; ++group) {
                int8_t * tile =
                    packed +
                    ((block * k4 + kb) * groups + group) * 16;
                for (int64_t ni = 0; ni < 4; ++ni) {
                    for (int64_t ki = 0; ki < 4; ++ki) {
                        const int64_t row =
                            block * block_n + group * 4 + ni;
                        tile[ni * 4 + ki] =
                            weight_nk[row * k + kb * 4 + ki];
                    }
                }
            }
        }
    }
    return 0;
}

extern "C" int triton_w8_launch_f32(
    triton_w8_kernel * kernel, const float * x, const int8_t * packed,
    const float * weight_scale, float * output) {
    if (!kernel || !x || !packed || !weight_scale || !output) {
        return fail("invalid arguments to triton_w8_launch_f32");
    }
    try {
        void * x_ptr = const_cast<float *>(x);
        void * packed_ptr = const_cast<int8_t *>(packed);
        void * scale_ptr = const_cast<float *>(weight_scale);
        void * output_ptr = output;
        void * args[] = {&x_ptr, &packed_ptr, &scale_ptr, &output_ptr};
        kernel->wrapper.launch_with_signature(
            static_cast<unsigned int>(kernel->grid_x), 1, 1, 1, nullptr,
            args, "*fp32,*i8,*fp32,*fp32", 4);
        return 0;
    } catch (const std::exception & error) {
        return fail(error);
    }
}

extern "C" int triton_w8_launch_f32_range(
    triton_w8_kernel * kernel, const float * x, const int8_t * packed,
    const float * weight_scale, float * output, int64_t block_start,
    int64_t block_count) {
    if (!kernel || !x || !packed || !weight_scale || !output ||
        block_start < 0 || block_count < 0 ||
        block_start + block_count > kernel->grid_x) {
        return fail("invalid arguments to triton_w8_launch_f32_range");
    }
    for (int64_t pid = block_start; pid < block_start + block_count; ++pid) {
        kernel->direct(
            const_cast<float *>(x), const_cast<int8_t *>(packed),
            const_cast<float *>(weight_scale), output,
            static_cast<uint32_t>(pid), 0, 0,
            static_cast<uint32_t>(kernel->grid_x), 1, 1);
    }
    return 0;
}

extern "C" triton_w8_split_kernel * triton_w8_split_kernel_create(
    const char * quant_kernel_dir, const char * gemv_kernel_dir,
    int64_t k, int64_t n, int64_t block_n) {
    try {
        if (!quant_kernel_dir || !gemv_kernel_dir ||
            !valid_config(k, n, block_n)) {
            fail("invalid Triton split W8 kernel configuration");
            return nullptr;
        }
        return new triton_w8_split_kernel(
            quant_kernel_dir, gemv_kernel_dir, k, n, block_n);
    } catch (const std::exception & error) {
        fail(error);
        return nullptr;
    }
}

extern "C" triton_w8_split_kernel *
triton_w8_split_kernel_create_from_bundle(
    const char * bundle_dir, int64_t k, int64_t n, int64_t block_n) {
    if (!bundle_dir) {
        fail("bundle directory is null");
        return nullptr;
    }
    const std::string kernel_dir =
        std::string(bundle_dir) + "/k" + std::to_string(k) + "-n" +
        std::to_string(n) + "-bn" + std::to_string(block_n);
    return triton_w8_split_kernel_create(
        kernel_dir.c_str(), kernel_dir.c_str(), k, n, block_n);
}

extern "C" void triton_w8_split_kernel_destroy(
    triton_w8_split_kernel * kernel) {
    delete kernel;
}

extern "C" int triton_w8_split_quantize_f32(
    triton_w8_split_kernel * kernel, const float * x) {
    if (!kernel || !x) {
        return fail("invalid arguments to triton_w8_split_quantize_f32");
    }
    kernel->quant(
        const_cast<float *>(x), kernel->x_q.data(), kernel->x_scale.data(),
        0, 0, 0, 1, 1, 1);
    return 0;
}

extern "C" int triton_w8_split_launch_prequant_f32_range(
    triton_w8_split_kernel * kernel, const int8_t * packed,
    const float * weight_scale, float * output,
    int64_t block_start, int64_t block_count) {
    if (!kernel || !packed || !weight_scale || !output || block_start < 0 ||
        block_count < 0 || block_start + block_count > kernel->grid_x) {
        return fail(
            "invalid arguments to "
            "triton_w8_split_launch_prequant_f32_range");
    }
    for (int64_t pid = block_start; pid < block_start + block_count; ++pid) {
        kernel->gemv(
            kernel->x_q.data(), kernel->x_scale.data(),
            const_cast<int8_t *>(packed),
            const_cast<float *>(weight_scale), output,
            static_cast<uint32_t>(pid), 0, 0,
            static_cast<uint32_t>(kernel->grid_x), 1, 1);
    }
    return 0;
}

extern "C" int triton_w8_split_launch_f32(
    triton_w8_split_kernel * kernel, const float * x,
    const int8_t * packed, const float * weight_scale, float * output) {
    if (triton_w8_split_quantize_f32(kernel, x) != 0) return -1;
    return triton_w8_split_launch_prequant_f32_range(
        kernel, packed, weight_scale, output, 0, kernel->grid_x);
}

extern "C" const char * triton_w8_last_error(void) {
    return last_error.c_str();
}
