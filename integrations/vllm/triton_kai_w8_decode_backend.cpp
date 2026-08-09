#include "triton_kai_w8_decode_backend.h"

#include <dlfcn.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using PackKernel = void (*)(void *, void *, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t);
using MatrixKernel = void (*)(void *, void *, void *, void *, int32_t, int32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t);
using PrefillMatrixKernel = void (*)(void *, void *, void *, void *, int32_t,
                                     int32_t, int32_t, int32_t, int32_t,
                                     int32_t);

thread_local std::string last_error;
thread_local std::vector<uint8_t> thread_lhs;

int fail(const char *message) {
  last_error = message;
  return -1;
}

int fail(const std::exception &error) {
  last_error = error.what();
  return -1;
}

class SharedObject {
public:
  explicit SharedObject(const std::string &path) {
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_)
      throw std::runtime_error(dlerror());
  }
  SharedObject(const SharedObject &) = delete;
  SharedObject &operator=(const SharedObject &) = delete;
  ~SharedObject() { dlclose(handle_); }

  template <typename Function> Function symbol(const char *name) {
    dlerror();
    void *address = dlsym(handle_, name);
    if (const char *error = dlerror())
      throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }

private:
  void *handle_ = nullptr;
};

bool valid_shape(int64_t k, int64_t n) {
  return k > 0 && n > 0 && k % 32 == 0 && n % 4 == 0;
}

int parallel_thread_count() {
#if defined(_OPENMP)
  return omp_get_num_threads();
#else
  return 1;
#endif
}

int parallel_thread_index() {
#if defined(_OPENMP)
  return omp_get_thread_num();
#else
  return 0;
#endif
}

int parallel_max_threads() {
#if defined(_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

} // namespace

extern "C" uint32_t triton_kai_w8_backend_abi_version(void) { return 2; }

struct triton_kai_w8_decode_kernel {
  triton_kai_w8_decode_kernel(std::string bundle, int64_t k_value,
                              int64_t n_value)
      : k(k_value), n(n_value), output_blocks(n_value / 4),
        shape_dir(std::move(bundle) + "/k" + std::to_string(k_value) + "-n" +
                  std::to_string(n_value)),
        pack_library(shape_dir + "/_pack_lhs_qai8dxp_bf16_kernel.so"),
        matrix_library(shape_dir + "/_kai_w8_layout_pointer_kernel.so"),
        pack_m16_library(
            shape_dir + "/_pack_lhs_qai8dxp_bf16_mr4_kernel.so"),
        prefill_m4_library(shape_dir + "/_kai_w8_prefill_m4_kernel.so"),
        prefill_m8_library(shape_dir + "/_kai_w8_prefill_m8_kernel.so"),
        prefill_m12_library(shape_dir + "/_kai_w8_prefill_m12_kernel.so"),
        prefill_m16_library(shape_dir + "/_kai_w8_prefill_kernel.so") {
    pack = pack_library.symbol<PackKernel>(
        "_pack_lhs_qai8dxp_bf16_kernel");
    matrix = matrix_library.symbol<MatrixKernel>(
        "_kai_w8_layout_pointer_kernel");
    pack_m16 = pack_m16_library.symbol<PackKernel>(
        "_pack_lhs_qai8dxp_bf16_mr4_kernel");
    prefill_m4 = prefill_m4_library.symbol<PrefillMatrixKernel>(
        "_kai_w8_prefill_short_tail_kernel");
    prefill_m8 = prefill_m8_library.symbol<PrefillMatrixKernel>(
        "_kai_w8_prefill_short_tail_kernel");
    prefill_m12 = prefill_m12_library.symbol<PrefillMatrixKernel>(
        "_kai_w8_prefill_m12_tail_kernel");
    prefill_m16 = prefill_m16_library.symbol<PrefillMatrixKernel>(
        "_kai_w8_prefill_kernel");
  }

  int64_t k;
  int64_t n;
  int32_t output_blocks;
  std::string shape_dir;
  SharedObject pack_library;
  SharedObject matrix_library;
  SharedObject pack_m16_library;
  SharedObject prefill_m4_library;
  SharedObject prefill_m8_library;
  SharedObject prefill_m12_library;
  SharedObject prefill_m16_library;
  PackKernel pack = nullptr;
  MatrixKernel matrix = nullptr;
  PackKernel pack_m16 = nullptr;
  PrefillMatrixKernel prefill_m4 = nullptr;
  PrefillMatrixKernel prefill_m8 = nullptr;
  PrefillMatrixKernel prefill_m12 = nullptr;
  PrefillMatrixKernel prefill_m16 = nullptr;
};

extern "C" triton_kai_w8_decode_kernel *
triton_kai_w8_decode_kernel_create(const char *bundle_dir, int64_t k,
                                   int64_t n) {
  try {
    if (!bundle_dir || !valid_shape(k, n)) {
      fail("invalid exact-KAI W8 decode configuration");
      return nullptr;
    }
    return new triton_kai_w8_decode_kernel(bundle_dir, k, n);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" void triton_kai_w8_decode_kernel_destroy(
    triton_kai_w8_decode_kernel *kernel) {
  delete kernel;
}

extern "C" int triton_kai_w8_decode_launch(
    triton_kai_w8_decode_kernel *kernel, const uint16_t *x_bf16,
    const uint8_t *rhs_packed, uint16_t *output_bf16) {
  if (!kernel || !x_bf16 || !rhs_packed || !output_bf16)
    return fail("invalid exact-KAI W8 decode arguments");
  try {
    const size_t lhs_size = static_cast<size_t>(kernel->k + 8);
    if (thread_lhs.size() < lhs_size)
      thread_lhs.resize(lhs_size);
    uint8_t *const lhs_packed = thread_lhs.data();
    kernel->pack(const_cast<uint16_t *>(x_bf16), lhs_packed, 0, 0, 0, 1, 1,
                 1);
    const float clamp[2] = {-std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max()};
    if (parallel_max_threads() == 1) {
      kernel->matrix(lhs_packed, const_cast<uint8_t *>(rhs_packed),
                     const_cast<float *>(clamp), output_bf16, 0,
                     kernel->output_blocks, 0, 0, 0, 1, 1, 1);
      return 0;
    }
#pragma omp parallel
    {
      const int32_t threads = parallel_thread_count();
      const int32_t thread = parallel_thread_index();
      const int32_t blocks_per_thread =
          (kernel->output_blocks + threads - 1) / threads;
      const int32_t begin = thread * blocks_per_thread;
      const int32_t end =
          std::min(begin + blocks_per_thread, kernel->output_blocks);
      if (begin < end)
        kernel->matrix(lhs_packed, const_cast<uint8_t *>(rhs_packed),
                       const_cast<float *>(clamp), output_bf16, begin, end, 0,
                       0, 0, 1, 1, 1);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" int triton_kai_w8_prefill_launch(
    triton_kai_w8_decode_kernel *kernel, int64_t m,
    const uint16_t *x_bf16, const uint8_t *rhs_packed,
    uint16_t *output_bf16) {
  if (!kernel || !x_bf16 || !rhs_packed || !output_bf16)
    return fail("invalid exact-KAI W8 prefill arguments");
  PrefillMatrixKernel prefill = nullptr;
  switch (m) {
  case 4:
    prefill = kernel->prefill_m4;
    break;
  case 8:
    prefill = kernel->prefill_m8;
    break;
  case 12:
    prefill = kernel->prefill_m12;
    break;
  case 16:
    prefill = kernel->prefill_m16;
    break;
  default:
    return fail("exact-KAI W8 prefill requires M in {4,8,12,16}");
  }
  try {
    const int32_t pack_programs = static_cast<int32_t>(m / 4);
    const size_t lhs_size = static_cast<size_t>(m) * (kernel->k + 8);
    if (thread_lhs.size() < lhs_size)
      thread_lhs.resize(lhs_size);
    uint8_t *const lhs_packed = thread_lhs.data();
    const float clamp[2] = {-std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max()};
    if (parallel_max_threads() == 1) {
      for (int32_t program = 0; program < pack_programs; ++program)
        kernel->pack_m16(const_cast<uint16_t *>(x_bf16), lhs_packed, program,
                         0, 0, pack_programs, 1, 1);
      for (int32_t pid_n = 0; pid_n < kernel->output_blocks; ++pid_n)
        prefill(
            lhs_packed, const_cast<uint8_t *>(rhs_packed),
            const_cast<float *>(clamp), output_bf16, 0, pid_n, 0, 1,
            kernel->output_blocks, 1);
      return 0;
    }
#pragma omp parallel
    {
      const int32_t threads = parallel_thread_count();
      const int32_t thread = parallel_thread_index();
      for (int32_t program = thread; program < pack_programs;
           program += threads)
        kernel->pack_m16(const_cast<uint16_t *>(x_bf16), lhs_packed,
                         program, 0, 0, pack_programs, 1, 1);
#pragma omp barrier
      const int32_t blocks_per_thread =
          (kernel->output_blocks + threads - 1) / threads;
      const int32_t begin = thread * blocks_per_thread;
      const int32_t end =
          std::min(begin + blocks_per_thread, kernel->output_blocks);
      for (int32_t pid_n = begin; pid_n < end; ++pid_n)
        prefill(
            lhs_packed, const_cast<uint8_t *>(rhs_packed),
            const_cast<float *>(clamp), output_bf16, 0, pid_n, 0, 1,
            kernel->output_blocks, 1);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" const char *triton_kai_w8_decode_last_error(void) {
  return last_error.c_str();
}
