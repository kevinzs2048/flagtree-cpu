#include "triton_p0_norm_backend.h"

#include <dlfcn.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using NormKernel = void (*)(void *, void *, void *, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t);
using RopeKernel = void (*)(void *, void *, void *, void *, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

thread_local std::string last_error;

int fail(const char *message) {
  last_error = message;
  return -1;
}

int fail(const std::exception &error) {
  last_error = error.what();
  return -1;
}

} // namespace

struct triton_p0_norm_kernel {
  triton_p0_norm_kernel(const char *kernel_dir, bool fused, int64_t rows_value)
      : rows(rows_value), fused_add(fused) {
    const char *symbol = fused ? "_vllm_fused_add_rms_aot_kernel"
                               : "_rms_norm_aot_kernel";
    const std::string path =
        std::string(kernel_dir) + "/" + symbol + ".so";
    module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module)
      throw std::runtime_error("failed to load generated norm kernel: " +
                               std::string(dlerror()));
    dlerror();
    function = reinterpret_cast<NormKernel>(dlsym(module, symbol));
    if (const char *error = dlerror()) {
      dlclose(module);
      module = nullptr;
      throw std::runtime_error("failed to find generated norm symbol: " +
                               std::string(error));
    }
  }

  ~triton_p0_norm_kernel() {
    if (module)
      dlclose(module);
  }

  void *module = nullptr;
  NormKernel function = nullptr;
  int64_t rows = 0;
  bool fused_add = false;
};

struct triton_p0_rope_kernel {
  triton_p0_rope_kernel(const char *kernel_dir, int64_t heads_value)
      : total_heads(heads_value) {
    constexpr const char *symbol = "_rope_qk_aot_kernel";
    const std::string path =
        std::string(kernel_dir) + "/" + symbol + ".so";
    module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module)
      throw std::runtime_error("failed to load generated RoPE kernel: " +
                               std::string(dlerror()));
    dlerror();
    function = reinterpret_cast<RopeKernel>(dlsym(module, symbol));
    if (const char *error = dlerror()) {
      dlclose(module);
      module = nullptr;
      throw std::runtime_error("failed to find generated RoPE symbol: " +
                               std::string(error));
    }
  }

  ~triton_p0_rope_kernel() {
    if (module)
      dlclose(module);
  }

  void *module = nullptr;
  RopeKernel function = nullptr;
  int64_t total_heads = 0;
};

extern "C" uint32_t triton_p0_norm_backend_abi_version(void) { return 2; }

extern "C" triton_p0_norm_kernel *triton_p0_norm_kernel_create(
    const char *kernel_dir, int32_t fused_add, int64_t rows) {
  try {
    if (!kernel_dir || (fused_add != 0 && fused_add != 1) || rows <= 0) {
      fail("invalid generated norm kernel configuration");
      return nullptr;
    }
    return new triton_p0_norm_kernel(kernel_dir, fused_add != 0, rows);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" void
triton_p0_norm_kernel_destroy(triton_p0_norm_kernel *kernel) {
  delete kernel;
}

extern "C" int triton_p0_rms_launch(triton_p0_norm_kernel *kernel,
                                     const uint16_t *input_bf16,
                                     const uint16_t *weight_bf16,
                                     uint16_t *output_bf16) {
  if (!kernel || kernel->fused_add || !input_bf16 || !weight_bf16 ||
      !output_bf16)
    return fail("invalid arguments to generated RMSNorm launch");
  try {
    for (uint32_t row = 0; row < static_cast<uint32_t>(kernel->rows); ++row)
      kernel->function(const_cast<uint16_t *>(input_bf16),
                       const_cast<uint16_t *>(weight_bf16), output_bf16, row,
                       0, 0, static_cast<uint32_t>(kernel->rows), 1, 1);
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" int triton_p0_fused_add_rms_launch(
    triton_p0_norm_kernel *kernel, uint16_t *input_bf16,
    uint16_t *residual_bf16, const uint16_t *weight_bf16) {
  if (!kernel || !kernel->fused_add || !input_bf16 || !residual_bf16 ||
      !weight_bf16)
    return fail("invalid arguments to generated fused RMSNorm launch");
  try {
    for (uint32_t row = 0; row < static_cast<uint32_t>(kernel->rows); ++row)
      kernel->function(input_bf16, residual_bf16,
                       const_cast<uint16_t *>(weight_bf16), row, 0, 0,
                       static_cast<uint32_t>(kernel->rows), 1, 1);
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" const char *triton_p0_norm_last_error(void) {
  return last_error.c_str();
}

extern "C" triton_p0_rope_kernel *
triton_p0_rope_kernel_create(const char *kernel_dir, int64_t total_heads) {
  try {
    if (!kernel_dir || total_heads <= 0) {
      fail("invalid generated RoPE kernel configuration");
      return nullptr;
    }
    return new triton_p0_rope_kernel(kernel_dir, total_heads);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" void
triton_p0_rope_kernel_destroy(triton_p0_rope_kernel *kernel) {
  delete kernel;
}

extern "C" int triton_p0_rope_launch(
    triton_p0_rope_kernel *kernel, uint16_t *query_bf16,
    uint16_t *key_bf16, const int64_t *positions,
    const uint16_t *cos_sin_cache_bf16) {
  if (!kernel || !query_bf16 || !key_bf16 || !positions ||
      !cos_sin_cache_bf16)
    return fail("invalid arguments to generated RoPE launch");
  try {
    for (uint32_t head = 0;
         head < static_cast<uint32_t>(kernel->total_heads); ++head)
      kernel->function(query_bf16, key_bf16,
                       const_cast<int64_t *>(positions),
                       const_cast<uint16_t *>(cos_sin_cache_bf16), head, 0, 0,
                       static_cast<uint32_t>(kernel->total_heads), 1, 1);
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}
