#include <arm_neon.h>

#include <stdint.h>

#if defined(SAME_BACKEND_NO_RESTRICT)
#define SAME_BACKEND_RESTRICT
#else
#define SAME_BACKEND_RESTRICT __restrict
#endif

typedef struct {
  float32x4_t lo;
  float32x4_t hi;
} f32x8;

static __attribute__((always_inline)) f32x8 load_bf16(const uint16_t *pointer) {
  const bfloat16x8_t packed =
      vld1q_bf16((const bfloat16_t *)pointer);
  f32x8 result = {vcvt_f32_bf16(vget_low_bf16(packed)),
                  vcvt_f32_bf16(vget_high_bf16(packed))};
  return result;
}

static __attribute__((always_inline)) void store_bf16(uint16_t *pointer,
                                                       f32x8 value) {
  vst1q_bf16((bfloat16_t *)pointer,
             vcombine_bf16(vcvt_bf16_f32(value.lo),
                           vcvt_bf16_f32(value.hi)));
}

static __attribute__((always_inline)) f32x8 multiply_add_rope(
    f32x8 first, f32x8 second, f32x8 cosine, f32x8 sine, int subtract) {
  f32x8 result;
  if (subtract) {
    result.lo = vfmsq_f32(vmulq_f32(first.lo, cosine.lo), second.lo, sine.lo);
    result.hi = vfmsq_f32(vmulq_f32(first.hi, cosine.hi), second.hi, sine.hi);
  } else {
    result.lo = vfmaq_f32(vmulq_f32(first.lo, sine.lo), second.lo, cosine.lo);
    result.hi = vfmaq_f32(vmulq_f32(first.hi, sine.hi), second.hi, cosine.hi);
  }
  return result;
}

static __attribute__((always_inline)) float tile_square_sum(f32x8 a,
                                                             f32x8 b) {
  float32x4_t sum = vmulq_f32(a.lo, a.lo);
  sum = vfmaq_f32(sum, a.hi, a.hi);
  sum = vfmaq_f32(sum, b.lo, b.lo);
  sum = vfmaq_f32(sum, b.hi, b.hi);
  return vaddvq_f32(sum);
}

__attribute__((visibility("default"))) void _rms_same_backend_acle(
    const uint16_t *SAME_BACKEND_RESTRICT input,
    const uint16_t *SAME_BACKEND_RESTRICT weight,
    uint16_t *SAME_BACKEND_RESTRICT output, uint32_t pid_x, uint32_t pid_y,
    uint32_t pid_z, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z) {
  (void)pid_y;
  (void)pid_z;
  (void)pid_x;
  (void)grid_x;
  (void)grid_y;
  (void)grid_z;
  const int size = 1024;
  float square_sum = 0.0f;
  for (int offset = 0; offset < size; offset += 16)
    square_sum += tile_square_sum(load_bf16(input + offset),
                                  load_bf16(input + offset + 8));
  const float rrms =
      1.0f / __builtin_sqrtf(square_sum / size + 1.0e-6f);
  const float32x4_t scale = vdupq_n_f32(rrms);
  for (int offset = 0; offset < size; offset += 8) {
    f32x8 value = load_bf16(input + offset);
    const f32x8 w = load_bf16(weight + offset);
    value.lo = vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(value.lo, scale)));
    value.hi = vcvt_f32_bf16(vcvt_bf16_f32(vmulq_f32(value.hi, scale)));
    f32x8 result = {vmulq_f32(value.lo, w.lo),
                    vmulq_f32(value.hi, w.hi)};
    store_bf16(output + offset, result);
  }
}

__attribute__((visibility("default"))) void _rope_same_backend_acle(
    uint16_t *SAME_BACKEND_RESTRICT q,
    uint16_t *SAME_BACKEND_RESTRICT k,
    const int64_t *SAME_BACKEND_RESTRICT positions,
    const uint16_t *SAME_BACKEND_RESTRICT cos_sin_cache, uint32_t pid_x,
    uint32_t pid_y,
    uint32_t pid_z, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z) {
  (void)pid_y;
  (void)pid_z;
  (void)grid_x;
  (void)grid_y;
  (void)grid_z;
  const int q_heads = 16;
  const int head_dim = 128;
  const int half = 64;
  uint16_t *row = pid_x < (uint32_t)q_heads
                      ? q + pid_x * head_dim
                      : k + (pid_x - q_heads) * head_dim;
  const uint16_t *cache = cos_sin_cache + positions[0] * head_dim;
#pragma clang loop unroll(full)
  for (int offset = 0; offset < half; offset += 8) {
    const f32x8 first = load_bf16(row + offset);
    const f32x8 second = load_bf16(row + half + offset);
    const f32x8 cosine = load_bf16(cache + offset);
    const f32x8 sine = load_bf16(cache + half + offset);
    store_bf16(row + offset,
               multiply_add_rope(first, second, cosine, sine, 1));
    store_bf16(row + half + offset,
               multiply_add_rope(first, second, cosine, sine, 0));
  }
}

static __attribute__((always_inline)) int32_t load_i32(const void *pointer) {
  int32_t value;
  __builtin_memcpy(&value, pointer, sizeof(value));
  return value;
}

static __attribute__((always_inline)) float load_f32(const void *pointer) {
  float value;
  __builtin_memcpy(&value, pointer, sizeof(value));
  return value;
}

__attribute__((visibility("default"))) void _w8_same_backend_acle(
    const int8_t *SAME_BACKEND_RESTRICT lhs_packed,
    const int8_t *SAME_BACKEND_RESTRICT rhs_packed,
    const float *SAME_BACKEND_RESTRICT clamp,
    float *SAME_BACKEND_RESTRICT output, int range_begin, int range_end,
    uint32_t pid_x,
    uint32_t pid_y, uint32_t pid_z, uint32_t grid_x, uint32_t grid_y,
    uint32_t grid_z) {
  (void)pid_x;
  (void)pid_y;
  (void)pid_z;
  (void)grid_x;
  (void)grid_y;
  (void)grid_z;
  const int k = 1024;
  const int rhs_stride = 4 * k + 48;
  const int lhs_offset = load_i32(lhs_packed + k);
  const float lhs_scale = load_f32(lhs_packed + k + 4);
  for (int block = range_begin; block < range_end; ++block) {
    const int8_t *rhs = rhs_packed + block * rhs_stride;
    int32x4_t partial01 = vdupq_n_s32(0);
    int32x4_t partial23 = vdupq_n_s32(0);
#pragma clang loop unroll_count(16)
    for (int group = 0; group < k / 8; ++group) {
      const int8x16_t x = vreinterpretq_s8_s64(
          vld1q_dup_s64((const int64_t *)(lhs_packed + group * 8)));
      partial01 = vdotq_s32(partial01, vld1q_s8(rhs + group * 32), x);
      partial23 =
          vdotq_s32(partial23, vld1q_s8(rhs + group * 32 + 16), x);
    }
    const int32x4_t dot = vpaddq_s32(partial01, partial23);
    const int8_t *epilogue = rhs + 4 * k;
    const int32x4_t corrected =
        vmlaq_n_s32(dot, vld1q_s32((const int32_t *)epilogue), lhs_offset);
    const float32x4_t scaled =
        vmulq_n_f32(vcvtq_f32_s32(corrected), lhs_scale);
    float32x4_t result =
        vfmaq_f32(vld1q_f32((const float *)(epilogue + 32)),
                   vld1q_f32((const float *)(epilogue + 16)), scaled);
    result = vmaxq_f32(result, vdupq_n_f32(clamp[0]));
    result = vminq_f32(result, vdupq_n_f32(clamp[1]));
    vst1q_f32(output + block * 4, result);
  }
}
