#include "triton_bf16_w8_backend.h"

#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "triton_jit/backends/cpu_backend.h"

namespace {

using QuantKernel = void (*)(void *, void *, void *, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, uint32_t);
using GemvKernel = void (*)(void *, void *, void *, void *, void *, int32_t,
                            int32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t);
using ActivationKernel = void (*)(void *, void *, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, uint32_t);
using RmsKernel = void (*)(void *, void *, void *, uint32_t, uint32_t,
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

bool valid_shape(int64_t k, int64_t n) {
  return k > 0 && n > 0 && k % 4 == 0 && n % 4 == 0;
}

int configured_vocab_threads(int64_t n) {
  if (n < 100000)
    return 1;
  const char *value = std::getenv("FLAGGEMS_ARM_W8_AOT_VOCAB_THREADS");
  if (!value || !*value)
    return 1;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (*end || parsed < 1 || parsed > 32)
    throw std::runtime_error(
        "FLAGGEMS_ARM_W8_AOT_VOCAB_THREADS must be in [1,32]");
  return static_cast<int>(parsed);
}

// This is only a persistent launch partitioner. Every worker invokes a
// disjoint range of the same compiler-generated Triton kernel; no operator
// arithmetic is implemented in the wrapper.
class ParallelGemv {
public:
  ParallelGemv(GemvKernel kernel, int workers)
      : kernel_(kernel), worker_count_(workers), start_(workers + 1),
        finish_(workers + 1) {
    for (int worker = 0; worker < worker_count_; ++worker)
      workers_.emplace_back([this, worker] { work(worker); });
  }

  ParallelGemv(const ParallelGemv &) = delete;
  ParallelGemv &operator=(const ParallelGemv &) = delete;

  ~ParallelGemv() {
    stop_ = true;
    start_.arrive_and_wait();
    for (auto &worker : workers_)
      worker.join();
  }

  void launch(void *x, void *x_scale, void *packed_weight,
              void *weight_scale, void *output, int32_t range_begin,
              int32_t range_end) {
    x_ = x;
    x_scale_ = x_scale;
    packed_weight_ = packed_weight;
    weight_scale_ = weight_scale;
    output_ = output;
    range_begin_ = range_begin;
    range_end_ = range_end;
    start_.arrive_and_wait();
    finish_.arrive_and_wait();
  }

private:
  void work(int worker) {
    while (true) {
      start_.arrive_and_wait();
      if (stop_)
        return;
      const int32_t range = range_end_ - range_begin_;
      const int32_t begin =
          range_begin_ + range * worker / worker_count_;
      const int32_t end =
          range_begin_ + range * (worker + 1) / worker_count_;
      kernel_(x_, x_scale_, packed_weight_, weight_scale_, output_, begin,
              end, 0, 0, 0, 1, 1, 1);
      finish_.arrive_and_wait();
    }
  }

  GemvKernel kernel_;
  int worker_count_;
  std::barrier<> start_;
  std::barrier<> finish_;
  std::vector<std::thread> workers_;
  bool stop_ = false;
  void *x_ = nullptr;
  void *x_scale_ = nullptr;
  void *packed_weight_ = nullptr;
  void *weight_scale_ = nullptr;
  void *output_ = nullptr;
  int32_t range_begin_ = 0;
  int32_t range_end_ = 0;
};

} // namespace

struct triton_bf16_w8_kernel {
  triton_bf16_w8_kernel(std::string quant_dir, std::string gemv_dir,
                        int64_t k_value, int64_t n_value,
                        int64_t block_n_value)
      : k(k_value), n(n_value), block_n(block_n_value),
        output_blocks(n_value / block_n_value),
        x_q(static_cast<size_t>(k_value)), x_scale(1) {
    auto quant_handle = triton_jit::CpuBackend::load_kernel(
        quant_dir, "_quantize_bf16_w8_kernel");
    auto gemv_handle = triton_jit::CpuBackend::load_kernel(
        gemv_dir, block_n == 4 ? "_w8a8_grouped_gemv_kernel"
                               : "_w8a8_wide_gemv_kernel");
    quant = reinterpret_cast<QuantKernel>(quant_handle.function);
    gemv = reinterpret_cast<GemvKernel>(gemv_handle.function);
    if (!quant || !gemv) {
      throw std::runtime_error("AOT bundle has no callable CPU symbol");
    }
    const int vocab_threads = configured_vocab_threads(n);
    if (vocab_threads > 1)
      parallel_gemv = std::make_unique<ParallelGemv>(gemv, vocab_threads);
  }

  int64_t k;
  int64_t n;
  int64_t block_n;
  int32_t output_blocks;
  std::vector<int8_t> x_q;
  std::vector<float> x_scale;
  QuantKernel quant = nullptr;
  GemvKernel gemv = nullptr;
  std::unique_ptr<ParallelGemv> parallel_gemv;
};

struct triton_bf16_w8_mlp_kernel {
  triton_bf16_w8_mlp_kernel(std::string quant_dir, std::string gemv_dir,
                            std::string activation_dir, int64_t k_value,
                            int64_t n_value, int64_t block_n_value)
      : k(k_value), n(n_value), block_n(block_n_value),
        output_blocks(2 * n_value / block_n_value),
        x_q(static_cast<size_t>(k_value)), x_scale(1),
        gate_up(static_cast<size_t>(2 * n_value)) {
    auto quant_handle = triton_jit::CpuBackend::load_kernel(
        quant_dir, "_quantize_bf16_w8_kernel");
    auto gemv_handle = triton_jit::CpuBackend::load_kernel(
        gemv_dir, "_w8a8_wide_gemv_kernel");
    const auto inline_activation =
        std::filesystem::path(activation_dir) /
        "_bf16_swiglu_inline_exp_kernel.so";
    const char *activation_name =
        std::filesystem::is_regular_file(inline_activation)
            ? "_bf16_swiglu_inline_exp_kernel"
            : "_bf16_swiglu_kernel";
    auto activation_handle = triton_jit::CpuBackend::load_kernel(
        activation_dir, activation_name);
    quant = reinterpret_cast<QuantKernel>(quant_handle.function);
    gemv = reinterpret_cast<GemvKernel>(gemv_handle.function);
    activation = reinterpret_cast<ActivationKernel>(activation_handle.function);
    if (!quant || !gemv || !activation) {
      throw std::runtime_error("AOT MLP bundle has no callable CPU symbol");
    }
  }

  int64_t k;
  int64_t n;
  int64_t block_n;
  int32_t output_blocks;
  std::vector<int8_t> x_q;
  std::vector<float> x_scale;
  std::vector<uint16_t> gate_up;
  QuantKernel quant = nullptr;
  GemvKernel gemv = nullptr;
  ActivationKernel activation = nullptr;
};

struct triton_bf16_rms_kernel {
  triton_bf16_rms_kernel(std::string kernel_dir, std::string kernel_name,
                         int64_t rows_value)
      : rows(rows_value) {
    auto kernel_handle = triton_jit::CpuBackend::load_kernel(
        kernel_dir, kernel_name);
    function = reinterpret_cast<RmsKernel>(kernel_handle.function);
    if (!function) {
      throw std::runtime_error("AOT RMSNorm bundle has no callable CPU symbol");
    }
  }

  int64_t rows;
  RmsKernel function = nullptr;
};

struct triton_bf16_rope_kernel {
  triton_bf16_rope_kernel(std::string kernel_dir, int64_t heads_value)
      : total_heads(heads_value) {
    auto kernel_handle = triton_jit::CpuBackend::load_kernel(
        kernel_dir, "_rope_qk_aot_kernel");
    function = reinterpret_cast<RopeKernel>(kernel_handle.function);
    if (!function) {
      throw std::runtime_error("AOT RoPE bundle has no callable CPU symbol");
    }
  }

  int64_t total_heads;
  RopeKernel function = nullptr;
};

extern "C" triton_bf16_w8_kernel *triton_bf16_w8_kernel_create(
    const char *quant_kernel_dir, const char *gemv_kernel_dir, int64_t k,
    int64_t n) {
  try {
    if (!quant_kernel_dir || !gemv_kernel_dir || !valid_shape(k, n)) {
      fail("invalid BF16-W8 kernel configuration");
      return nullptr;
    }
    return new triton_bf16_w8_kernel(
        quant_kernel_dir, gemv_kernel_dir, k, n, 4);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" triton_bf16_w8_kernel *triton_bf16_w8_kernel_create_wide(
    const char *quant_kernel_dir, const char *gemv_kernel_dir, int64_t k,
    int64_t n, int64_t block_n) {
  try {
    if (!quant_kernel_dir || !gemv_kernel_dir || !valid_shape(k, n) ||
        block_n <= 0 || block_n % 4 != 0 || n % block_n != 0) {
      fail("invalid wide BF16-W8 kernel configuration");
      return nullptr;
    }
    return new triton_bf16_w8_kernel(
        quant_kernel_dir, gemv_kernel_dir, k, n, block_n);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" void
triton_bf16_w8_kernel_destroy(triton_bf16_w8_kernel *kernel) {
  delete kernel;
}

extern "C" size_t triton_bf16_w8_packed_size(int64_t k, int64_t n) {
  if (!valid_shape(k, n)) {
    return 0;
  }
  return static_cast<size_t>(k) * static_cast<size_t>(n);
}

extern "C" int triton_bf16_w8_launch(triton_bf16_w8_kernel *kernel,
                                      const uint16_t *x_bf16,
                                      const int8_t *packed_weight,
                                      const float *weight_scale,
                                      uint16_t *output_bf16) {
  if (!kernel || !x_bf16 || !packed_weight || !weight_scale ||
      !output_bf16) {
    return fail("invalid arguments to triton_bf16_w8_launch");
  }
  try {
    kernel->quant(const_cast<uint16_t *>(x_bf16), kernel->x_q.data(),
                  kernel->x_scale.data(), 0, 0, 0, 1, 1, 1);
    if (kernel->parallel_gemv) {
      kernel->parallel_gemv->launch(
          kernel->x_q.data(), kernel->x_scale.data(),
          const_cast<int8_t *>(packed_weight),
          const_cast<float *>(weight_scale), output_bf16, 0,
          kernel->output_blocks);
    } else {
      kernel->gemv(kernel->x_q.data(), kernel->x_scale.data(),
                   const_cast<int8_t *>(packed_weight),
                   const_cast<float *>(weight_scale), output_bf16, 0,
                   kernel->output_blocks, 0, 0, 0, 1, 1, 1);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" triton_bf16_w8_mlp_kernel *triton_bf16_w8_mlp_kernel_create(
    const char *quant_kernel_dir, const char *gemv_kernel_dir,
    const char *activation_kernel_dir, int64_t k, int64_t n,
    int64_t block_n) {
  try {
    if (!quant_kernel_dir || !gemv_kernel_dir || !activation_kernel_dir ||
        !valid_shape(k, n) || block_n <= 0 || block_n % 4 != 0 ||
        (2 * n) % block_n != 0) {
      fail("invalid BF16-W8 MLP kernel configuration");
      return nullptr;
    }
    return new triton_bf16_w8_mlp_kernel(
        quant_kernel_dir, gemv_kernel_dir, activation_kernel_dir, k, n,
        block_n);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" void triton_bf16_w8_mlp_kernel_destroy(
    triton_bf16_w8_mlp_kernel *kernel) {
  delete kernel;
}

extern "C" int triton_bf16_w8_mlp_launch(
    triton_bf16_w8_mlp_kernel *kernel, const uint16_t *x_bf16,
    const int8_t *packed_gate_up, const float *gate_up_scale,
    uint16_t *output_bf16) {
  if (!kernel || !x_bf16 || !packed_gate_up || !gate_up_scale ||
      !output_bf16) {
    return fail("invalid arguments to triton_bf16_w8_mlp_launch");
  }
  try {
    kernel->quant(const_cast<uint16_t *>(x_bf16), kernel->x_q.data(),
                  kernel->x_scale.data(), 0, 0, 0, 1, 1, 1);
    kernel->gemv(kernel->x_q.data(), kernel->x_scale.data(),
                 const_cast<int8_t *>(packed_gate_up),
                 const_cast<float *>(gate_up_scale), kernel->gate_up.data(),
                 0, kernel->output_blocks, 0, 0, 0, 1, 1, 1);
    kernel->activation(kernel->gate_up.data(), output_bf16,
                       0, 0, 0, 1, 1, 1);
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" triton_bf16_rms_kernel *triton_bf16_rms_kernel_create(
    const char *kernel_dir, int64_t rows) {
  try {
    if (!kernel_dir || rows <= 0) {
      fail("invalid BF16 RMSNorm kernel configuration");
      return nullptr;
    }
    return new triton_bf16_rms_kernel(
        kernel_dir, "_rms_norm_aot_kernel", rows);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" triton_bf16_rms_kernel *
triton_bf16_fused_add_rms_kernel_create(const char *kernel_dir,
                                        int64_t rows) {
  try {
    if (!kernel_dir || rows <= 0) {
      fail("invalid BF16 fused-add RMSNorm kernel configuration");
      return nullptr;
    }
    return new triton_bf16_rms_kernel(
        kernel_dir, "_fused_add_rms_aot_kernel", rows);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" void triton_bf16_rms_kernel_destroy(
    triton_bf16_rms_kernel *kernel) {
  delete kernel;
}

extern "C" int triton_bf16_rms_launch(
    triton_bf16_rms_kernel *kernel, const uint16_t *x_bf16,
    const uint16_t *weight_bf16, uint16_t *output_bf16) {
  if (!kernel || !x_bf16 || !weight_bf16 || !output_bf16) {
    return fail("invalid arguments to triton_bf16_rms_launch");
  }
  try {
    for (uint32_t row = 0; row < kernel->rows; ++row) {
      kernel->function(const_cast<uint16_t *>(x_bf16),
                       const_cast<uint16_t *>(weight_bf16), output_bf16,
                       row, 0, 0, static_cast<uint32_t>(kernel->rows), 1, 1);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" int triton_bf16_fused_add_rms_launch(
    triton_bf16_rms_kernel *kernel, uint16_t *input_bf16,
    uint16_t *residual_bf16, const uint16_t *weight_bf16) {
  if (!kernel || !input_bf16 || !residual_bf16 || !weight_bf16) {
    return fail("invalid arguments to triton_bf16_fused_add_rms_launch");
  }
  try {
    for (uint32_t row = 0; row < kernel->rows; ++row) {
      kernel->function(input_bf16, residual_bf16,
                       const_cast<uint16_t *>(weight_bf16),
                       row, 0, 0, static_cast<uint32_t>(kernel->rows), 1, 1);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" triton_bf16_rope_kernel *triton_bf16_rope_kernel_create(
    const char *kernel_dir, int64_t total_heads) {
  try {
    if (!kernel_dir || total_heads <= 0) {
      fail("invalid BF16 RoPE kernel configuration");
      return nullptr;
    }
    return new triton_bf16_rope_kernel(kernel_dir, total_heads);
  } catch (const std::exception &error) {
    fail(error);
    return nullptr;
  }
}

extern "C" void triton_bf16_rope_kernel_destroy(
    triton_bf16_rope_kernel *kernel) {
  delete kernel;
}

extern "C" int triton_bf16_rope_launch(
    triton_bf16_rope_kernel *kernel, uint16_t *query_bf16,
    uint16_t *key_bf16, const uint16_t *cosine_bf16,
    const uint16_t *sine_bf16) {
  if (!kernel || !query_bf16 || !key_bf16 || !cosine_bf16 ||
      !sine_bf16) {
    return fail("invalid arguments to triton_bf16_rope_launch");
  }
  try {
    for (uint32_t head = 0; head < kernel->total_heads; ++head) {
      kernel->function(query_bf16, key_bf16,
                       const_cast<uint16_t *>(cosine_bf16),
                       const_cast<uint16_t *>(sine_bf16),
                       head, 0, 0,
                       static_cast<uint32_t>(kernel->total_heads), 1, 1);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(error);
  }
}

extern "C" const char *triton_bf16_w8_last_error(void) {
  return last_error.c_str();
}
