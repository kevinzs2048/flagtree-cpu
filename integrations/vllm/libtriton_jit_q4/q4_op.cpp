#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>

#include "triton_jit/triton_jit_function.h"

namespace {

using triton_jit::TritonJITFunction;

constexpr int64_t kBlockLength = 32;
constexpr int64_t kG128BlockLength = 128;

std::vector<int64_t> output_shape(const at::Tensor& input, int64_t n) {
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  TORCH_CHECK(!shape.empty(), "Q4 input must have at least one dimension");
  shape.back() = n;
  return shape;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0;
       --index) {
    strides[index] = strides[index + 1] * shape[index + 1];
  }
  return strides;
}

int32_t checked_i32(int64_t value, const char* name) {
  TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() &&
                  value <= std::numeric_limits<int32_t>::max(),
              name, " does not fit the Triton CPU i32 ABI");
  return static_cast<int32_t>(value);
}

int32_t decode_partitions(int64_t k, int64_t n) {
  if (k * n < 2 * 1024 * 1024) {
    return 1;
  }
  const int32_t threads = std::max(1, at::get_num_threads());
  if (const char* configured = std::getenv("FLAGGEMS_Q4_DECODE_PARTITIONS")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    TORCH_CHECK(end != configured && *end == '\0' && parsed > 0,
                "FLAGGEMS_Q4_DECODE_PARTITIONS must be a positive integer");
    return std::min<int32_t>(
        std::min<int32_t>(threads, checked_i32(parsed, "decode partitions")),
        checked_i32(n / 64, "N/64"));
  }
  return std::min<int32_t>(threads, checked_i32(n / 64, "N/64"));
}

int32_t decode_unroll(int64_t k, int64_t n) {
  const char* configured = std::getenv("FLAGGEMS_Q4_DECODE_UNROLL");
  if (configured == nullptr) {
    // The wide joined gate/up projection has enough work per partition to
    // amortize a two-group unroll.  Narrow projections and the K6144 down
    // projection are neutral or regress because the larger body competes
    // with activation-pack state and cache bandwidth.
    return n >= 8192 && k <= 2048 ? 2 : 1;
  }
  char* end = nullptr;
  const long parsed = std::strtol(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' &&
                  (parsed == 1 || parsed == 2 || parsed == 4),
              "FLAGGEMS_Q4_DECODE_UNROLL must be 1, 2, or 4");
  return static_cast<int32_t>(parsed);
}

int32_t tail_block(int64_t rows) {
  TORCH_CHECK(rows > 0 && rows <= 16, "invalid Q4 prefill tail: ", rows);
  return static_cast<int32_t>(std::min<int64_t>(16, 4 * ((rows + 3) / 4)));
}

int32_t g128_prefill_block(int64_t rows) {
  if (rows <= 16) {
    return 16;
  }
  if (rows >= 96 || rows % 12 == 0) {
    return 12;
  }
  if (rows % 16 == 0) {
    return 16;
  }
  if (rows % 8 == 0) {
    return 8;
  }
  return 12;
}

void validate(const at::Tensor& input,
              const at::Tensor& rhs,
              int64_t n,
              int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && rhs.device().is_cpu(),
              "libtriton_jit Q4 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "libtriton_jit Q4 requires BF16 activation input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "libtriton_jit Q4 requires a contiguous UINT8 RHS blob");
  TORCH_CHECK(input.is_contiguous(),
              "libtriton_jit Q4 requires contiguous activation input");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid Q4 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % kBlockLength == 0,
              "Q4 dimensions require N%4=0 and K%32=0");
  const int64_t expected = (n / 4) * (k / kBlockLength) * 72;
  TORCH_CHECK(rhs.numel() == expected, "invalid packed Q4 RHS byte count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

void validate_g128(const at::Tensor& input,
                   const at::Tensor& rhs,
                   int64_t n,
                   int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && rhs.device().is_cpu(),
              "libtriton_jit G128 Q4 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "libtriton_jit G128 Q4 requires BF16 activation input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "libtriton_jit G128 Q4 requires a contiguous UINT8 RHS blob");
  TORCH_CHECK(input.is_contiguous(),
              "libtriton_jit G128 Q4 requires contiguous activation input");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid G128 Q4 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % kG128BlockLength == 0,
              "G128 Q4 dimensions require N%4=0 and K%128=0");
  const int64_t expected =
      (n / 4) * ((k / kG128BlockLength) * 264 + 16);
  TORCH_CHECK(rhs.numel() == expected,
              "invalid packed G128 Q4 RHS byte count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

void validate_g32_asym(const at::Tensor& input,
                       const at::Tensor& rhs,
                       int64_t n,
                       int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && rhs.device().is_cpu(),
              "libtriton_jit asymmetric G32 Q4 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
              "asymmetric G32 Q4 requires contiguous BF16 input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "asymmetric G32 Q4 requires a contiguous UINT8 RHS blob");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid asymmetric G32 Q4 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % kBlockLength == 0,
              "asymmetric G32 Q4 dimensions require N%4=0 and K%32=0");
  const int64_t expected = (n / 4) * (k / kBlockLength) * 80;
  TORCH_CHECK(rhs.numel() == expected,
              "invalid packed asymmetric G32 Q4 RHS byte count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

at::Tensor run_decode(const at::Tensor& input,
                      const at::Tensor& rhs,
                      int64_t n,
                      int64_t k,
                      int64_t m) {
  const int32_t partitions = decode_partitions(k, n);
  const int64_t scratch_bytes =
      m * partitions * (k / kBlockLength) * 34;
  const int64_t output_bytes = m * n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned Q4 workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  const std::vector<int64_t> shape = output_shape(input, n);
  at::Tensor output = storage.as_strided(
      shape, contiguous_strides(shape), scratch_bytes / 2);

  TritonJITFunction& kernel = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_fused_decode_sdot_kai_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t output_offset = checked_i32(scratch_bytes, "workspace");
  const int32_t range_begin = 0;
  const int32_t range_end = n32 / 4;
  const int32_t unroll = k >= 4096 ? 4 : 1;
  kernel(nullptr,
         static_cast<unsigned int>(m32),
         static_cast<unsigned int>(partitions),
         1,
         1,
         1,
         input,
         storage,
         rhs,
         output_offset,
         k32,
         range_begin,
         range_end,
         k32,
         n32,
         unroll);
  return output;
}

at::Tensor run_prefill(const at::Tensor& input,
                       const at::Tensor& rhs,
                       int64_t n,
                       int64_t k,
                       int64_t m) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  const int64_t groups = k / kBlockLength;
  const int64_t lhs_bytes = (padded_m / 4) * groups * 136;
  at::Tensor lhs_blob = at::empty({lhs_bytes}, input.options().dtype(at::kByte));
  at::Tensor lhs_scale = lhs_blob.view(at::kHalf);
  at::Tensor lhs_data = lhs_blob.view(at::kChar);
  at::Tensor rhs_scale = rhs.view(at::kHalf);
  at::Tensor input_2d = input.view({m, k});

  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_pack_lhs_qsi8d32p_panel4_scalar_kernel");
  TritonJITFunction& matrix = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_i8mm_kai_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t full_panels = m32 / 4;
  if (full_panels > 0) {
    pack(nullptr,
         static_cast<unsigned int>(full_panels),
         1,
         1,
         1,
         1,
         input_2d,
         lhs_scale,
         lhs_data,
         m32,
         k32,
         k32,
         true);
  }
  const int32_t tail_rows = m32 - full_panels * 4;
  if (tail_rows > 0) {
    const int64_t lhs_offset = static_cast<int64_t>(full_panels) * groups * 136;
    at::Tensor input_tail = input_2d.narrow(0, full_panels * 4, tail_rows);
    at::Tensor lhs_tail = lhs_blob.narrow(0, lhs_offset, lhs_bytes - lhs_offset);
    pack(nullptr,
         1,
         1,
         1,
         1,
         1,
         input_tail,
         lhs_tail.view(at::kHalf),
         lhs_tail.view(at::kChar),
         tail_rows,
         k32,
         k32,
         false);
  }

  at::Tensor output = at::empty({padded_m, n}, input.options());
  const int32_t main_rows = (m32 / 16) * 16;
  const int32_t main_tiles = main_rows / 16;
  if (main_tiles > 0) {
    matrix(nullptr,
           static_cast<unsigned int>(main_tiles),
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs_data,
           lhs_scale,
           rhs,
           rhs_scale,
           output,
           n32,
           k32,
           16);
  }
  const int32_t remaining = m32 - main_rows;
  if (remaining > 0) {
    const int32_t block_m = tail_block(remaining);
    const int64_t lhs_offset = (static_cast<int64_t>(main_rows) / 4) * groups * 136;
    at::Tensor lhs_tail = lhs_blob.narrow(0, lhs_offset, lhs_bytes - lhs_offset);
    at::Tensor output_tail = output.narrow(0, main_rows, padded_m - main_rows);
    matrix(nullptr,
           1,
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs_tail.view(at::kChar),
           lhs_tail.view(at::kHalf),
           rhs,
           rhs_scale,
           output_tail,
           n32,
           k32,
           block_m);
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

at::Tensor q4_linear_cpu(const at::Tensor& input,
                         const at::Tensor& rhs,
                         int64_t n,
                         int64_t k) {
  validate(input, rhs, n, k);
  const int64_t m = input.numel() / k;
  return m < 4 ? run_decode(input, rhs, n, k, m)
               : run_prefill(input, rhs, n, k, m);
}

at::Tensor run_g32_asym_decode(const at::Tensor& input,
                               const at::Tensor& rhs,
                               int64_t n,
                               int64_t k,
                               int64_t m) {
  const int32_t partitions = decode_partitions(k, n);
  const int64_t scratch_bytes = m * partitions * (8 + k);
  const int64_t output_bytes = m * n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned asymmetric G32 Q4 workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  at::Tensor output = storage.as_strided(
      output_shape(input, n), contiguous_strides(output_shape(input, n)),
      scratch_bytes / 2);

  TritonJITFunction& kernel = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_fused_decode_asym_g32_kai_sdot_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  kernel(nullptr,
         static_cast<unsigned int>(m32),
         static_cast<unsigned int>(partitions),
         1,
         1,
         1,
         input,
         storage,
         rhs,
         checked_i32(scratch_bytes, "workspace"),
         k32,
         0,
         n32 / 4,
         k32,
         n32,
         decode_unroll(k, n));
  return output;
}

at::Tensor run_g32_asym_prefill(const at::Tensor& input,
                                const at::Tensor& rhs,
                                int64_t n,
                                int64_t k,
                                int64_t m) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  const int64_t lhs_panel_stride = 32 + 4 * k;
  const int64_t lhs_bytes = (padded_m / 4) * lhs_panel_stride;
  at::Tensor lhs = at::empty({lhs_bytes}, input.options().dtype(at::kByte));
  at::Tensor output = at::empty({padded_m, n}, input.options());
  at::Tensor input_2d = input.view({m, k});

  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_pack_lhs_qai8dxp_asym_panel4_kernel");
  TritonJITFunction& matrix = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_asym_i8mm_kai_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  pack(nullptr,
       static_cast<unsigned int>(padded_m / 4),
       1,
       1,
       1,
       1,
       input_2d,
       lhs,
       m32,
       k32,
       k32);

  // CIX A/B: one grid of paired M8 tiles beats the spilling M16 object,
  // whereas the joined M12 tail is faster than a separate M8 plus M4 launch.
  const int32_t main_block = m32 >= 16 ? 8 : 16;
  const int32_t main_rows = (m32 / main_block) * main_block;
  if (main_rows > 0) {
    matrix(nullptr,
           static_cast<unsigned int>(main_rows / main_block),
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs,
           rhs,
           output,
           n32,
           k32,
           main_block,
           true);
  }
  const int32_t remaining = m32 - main_rows;
  if (remaining > 0) {
    const int32_t block_m = tail_block(remaining);
    const int64_t lhs_offset =
        (static_cast<int64_t>(main_rows) / 4) * lhs_panel_stride;
    matrix(nullptr,
           1,
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs.narrow(0, lhs_offset, lhs_bytes - lhs_offset),
           rhs,
           output.narrow(0, main_rows, padded_m - main_rows),
           n32,
           k32,
           block_m,
           true);
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

at::Tensor q4_linear_g32_asym_cpu(const at::Tensor& input,
                                  const at::Tensor& rhs,
                                  int64_t n,
                                  int64_t k) {
  validate_g32_asym(input, rhs, n, k);
  const int64_t m = input.numel() / k;
  return m < 4 ? run_g32_asym_decode(input, rhs, n, k, m)
               : run_g32_asym_prefill(input, rhs, n, k, m);
}

at::Tensor run_g128_decode(const at::Tensor& input,
                           const at::Tensor& rhs,
                           int64_t n,
                           int64_t k,
                           int64_t m) {
  const int32_t partitions = decode_partitions(k, n);
  const int64_t scratch_bytes = m * partitions * (8 + k);
  const int64_t output_bytes = m * n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned G128 Q4 workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  const std::vector<int64_t> shape = output_shape(input, n);
  at::Tensor output = storage.as_strided(
      shape, contiguous_strides(shape), scratch_bytes / 2);

  TritonJITFunction& kernel = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_fused_decode_asym_g128_kai_sdot_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t output_offset = checked_i32(scratch_bytes, "workspace");
  const int32_t range_begin = 0;
  const int32_t range_end = n32 / 4;
  const int32_t unroll = decode_unroll(k, n);
  kernel(nullptr,
         static_cast<unsigned int>(m32),
         static_cast<unsigned int>(partitions),
         1,
         1,
         1,
         input,
         storage,
         rhs,
         output_offset,
         k32,
         range_begin,
         range_end,
         k32,
         n32,
         unroll);
  return output;
}

at::Tensor run_g128_prefill(const at::Tensor& input,
                            const at::Tensor& rhs,
                            int64_t n,
                            int64_t k,
                            int64_t m) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  const int64_t lhs_panel_stride = 32 + 4 * k;
  const int64_t lhs_bytes = (padded_m / 4) * lhs_panel_stride;
  at::Tensor lhs_blob = at::empty({lhs_bytes}, input.options().dtype(at::kByte));
  at::Tensor input_2d = input.view({m, k});

  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_pack_lhs_qai8dxp_asym_panel4_kernel");
  TritonJITFunction& matrix = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_asym_g128_i8mm_kernel");
  TritonJITFunction& matrix_m12_k32 = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_asym_g128_i8mm_kai_m12_k32_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t panels = checked_i32(padded_m / 4, "G128 panels");
  pack(nullptr,
       static_cast<unsigned int>(panels),
       1,
       1,
       1,
       1,
       input_2d,
       lhs_blob,
       m32,
       k32,
       k32);

  at::Tensor output = at::empty({padded_m, n}, input.options());
  const int32_t block_m = g128_prefill_block(m);
  const int32_t main_rows = (m32 / block_m) * block_m;
  const int32_t main_tiles = main_rows / block_m;
  if (main_tiles > 0) {
    matrix(nullptr,
           static_cast<unsigned int>(main_tiles),
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs_blob,
           rhs,
           output,
           n32,
           k32,
           block_m,
           true);
  }
  const int32_t remaining = m32 - main_rows;
  if (remaining > 0) {
    const int32_t tail = tail_block(remaining);
    const int64_t lhs_offset =
        (static_cast<int64_t>(main_rows) / 4) * lhs_panel_stride;
    at::Tensor lhs_tail = lhs_blob.narrow(0, lhs_offset, lhs_bytes - lhs_offset);
    at::Tensor output_tail = output.narrow(0, main_rows, padded_m - main_rows);
    if (tail == 12 && n >= 8192 && k <= 2048) {
      matrix_m12_k32(nullptr,
                     1,
                     static_cast<unsigned int>(n32 / 4),
                     1,
                     1,
                     1,
                     lhs_tail,
                     rhs,
                     output_tail,
                     n32,
                     k32);
    } else {
      matrix(nullptr,
             1,
             static_cast<unsigned int>(n32 / 4),
             1,
             1,
             1,
             lhs_tail,
             rhs,
             output_tail,
             n32,
             k32,
             tail,
             true);
    }
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

at::Tensor q4_linear_g128_cpu(const at::Tensor& input,
                              const at::Tensor& rhs,
                              int64_t n,
                              int64_t k) {
  validate_g128(input, rhs, n, k);
  const int64_t m = input.numel() / k;
  return m < 4 ? run_g128_decode(input, rhs, n, k, m)
               : run_g128_prefill(input, rhs, n, k, m);
}

at::Tensor q4_linear_meta(const at::Tensor& input,
                          const at::Tensor&,
                          int64_t n,
                          int64_t) {
  std::vector<c10::SymInt> shape = input.sym_sizes().vec();
  TORCH_CHECK(!shape.empty(), "Q4 input must have at least one dimension");
  shape.back() = c10::SymInt(n);
  return input.new_empty_symint(shape, input.options().dtype(at::kBFloat16));
}

}  // namespace

TORCH_LIBRARY(triton_jit_cpu, library) {
  library.def("q4_linear(Tensor input, Tensor rhs, int n, int k) -> Tensor");
  library.def(
      "q4_linear_g32_asym(Tensor input, Tensor rhs, int n, int k) -> Tensor");
  library.def(
      "q4_linear_g128(Tensor input, Tensor rhs, int n, int k) -> Tensor");
}

TORCH_LIBRARY_IMPL(triton_jit_cpu, CPU, library) {
  library.impl("q4_linear", q4_linear_cpu);
  library.impl("q4_linear_g32_asym", q4_linear_g32_asym_cpu);
  library.impl("q4_linear_g128", q4_linear_g128_cpu);
}

TORCH_LIBRARY_IMPL(triton_jit_cpu, Meta, library) {
  library.impl("q4_linear", q4_linear_meta);
  library.impl("q4_linear_g32_asym", q4_linear_meta);
  library.impl("q4_linear_g128", q4_linear_meta);
}
