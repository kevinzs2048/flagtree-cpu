#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline float bf16_to_float(uint16_t value) {
  uint32_t bits = (uint32_t)value << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static inline uint16_t float_to_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

static inline int32_t round_to_nearest_even(float value) {
  const int32_t base = (int32_t)value;
  const float fraction = fabsf(value - (float)base);
  const int32_t direction = value < 0.0f ? -1 : 1;
  if (fraction > 0.5f ||
      (fraction == 0.5f && (base & 1) != 0)) {
    return base + direction;
  }
  return base;
}

static float quantize_bf16(const uint16_t *restrict x,
                           int8_t *restrict x_q,
                           float *restrict x_scale, int k) {
  float absmax = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    const float value = fabsf(bf16_to_float(x[inner]));
    if (value > absmax) {
      absmax = value;
    }
  }
  if (absmax < 1.0e-8f) {
    absmax = 1.0e-8f;
  }
  const float scale = absmax / 127.0f;
  const float inv_scale = 127.0f / absmax;
  x_scale[0] = scale;
  for (int inner = 0; inner < k; ++inner) {
    float value = bf16_to_float(x[inner]) * inv_scale;
    if (value > 127.0f) {
      value = 127.0f;
    } else if (value < -128.0f) {
      value = -128.0f;
    }
    x_q[inner] = (int8_t)round_to_nearest_even(value);
  }
  return scale;
}

// Independent fused C/ACLE ceiling for the two ordinary-Triton stages in
// bench_bf16_w8a8_ordinary_split.py.  Packed weights use
// [N / 4, K / 4, 4 output lanes, 4 K lanes].
__attribute__((noinline, visibility("default")))
void bf16_w8a8_split_c(const uint16_t *restrict x,
                       int8_t *restrict x_q,
                       float *restrict x_scale,
                       const int8_t *restrict packed,
                       const float *restrict weight_scale,
                       uint16_t *restrict output, int k, int n) {
  const float scale = quantize_bf16(x, x_q, x_scale, k);

  const int k_groups = k / 4;
  const int tiles = n / 4;
  for (int tile = 0; tile < tiles; ++tile) {
    int32x4_t dot = vdupq_n_s32(0);
    const int8_t *tile_packed = packed + (size_t)tile * k_groups * 16;
    for (int group = 0; group < k_groups; ++group) {
      uint32_t x4;
      memcpy(&x4, x_q + group * 4, 4);
      const int8x16_t activation =
          vreinterpretq_s8_u32(vdupq_n_u32(x4));
      const int8x16_t weight =
          vld1q_s8(tile_packed + (size_t)group * 16);
      dot = vdotq_s32(dot, activation, weight);
    }
    float result[4];
    float32x4_t values = vmulq_n_f32(vcvtq_f32_s32(dot), scale);
    values = vmulq_f32(
        values, vld1q_f32(weight_scale + tile * 4));
    vst1q_f32(result, values);
    for (int lane = 0; lane < 4; ++lane) {
      output[tile * 4 + lane] = float_to_bf16(result[lane]);
    }
  }
}

// Same computation for [N/BLOCK_N,K/4,BLOCK_N,4] packed weights.  This is
// the layout consumed by the ordinary-Triton wide-output experiment.
static void bf16_w8a8_wide64_gemv(const int8_t *restrict x_q, float scale,
                                  const int8_t *restrict packed,
                                  const float *restrict weight_scale,
                                  uint16_t *restrict output, int k, int n) {
  const int k_groups = k / 4;
  const int blocks = n / 64;
  for (int block = 0; block < blocks; ++block) {
    int32x4_t acc[16];
#pragma GCC unroll 16
    for (int tile = 0; tile < 16; ++tile) {
      acc[tile] = vdupq_n_s32(0);
    }
    for (int group = 0; group < k_groups; ++group) {
      uint32_t x4;
      memcpy(&x4, x_q + group * 4, 4);
      const int8x16_t activation =
          vreinterpretq_s8_u32(vdupq_n_u32(x4));
      const int8_t *weight_base =
          packed + ((size_t)block * k_groups + group) * 64 * 4;
#pragma GCC unroll 16
      for (int tile = 0; tile < 16; ++tile) {
        acc[tile] = vdotq_s32(
            acc[tile], activation, vld1q_s8(weight_base + tile * 16));
      }
    }
#pragma GCC unroll 16
    for (int tile = 0; tile < 16; ++tile) {
      const int output_offset = block * 64 + tile * 4;
      float result[4];
      float32x4_t values =
          vmulq_n_f32(vcvtq_f32_s32(acc[tile]), scale);
      values = vmulq_f32(
          values, vld1q_f32(weight_scale + output_offset));
      vst1q_f32(result, values);
#pragma GCC unroll 4
      for (int lane = 0; lane < 4; ++lane) {
        output[output_offset + lane] = float_to_bf16(result[lane]);
      }
    }
  }
}

__attribute__((noinline, visibility("default")))
void bf16_w8a8_wide_c(const uint16_t *restrict x,
                      int8_t *restrict x_q,
                      float *restrict x_scale,
                      const int8_t *restrict packed,
                      const float *restrict weight_scale,
                      uint16_t *restrict output, int k, int n, int block_n) {
  const float scale = quantize_bf16(x, x_q, x_scale, k);
  if (block_n == 64) {
    bf16_w8a8_wide64_gemv(
        x_q, scale, packed, weight_scale, output, k, n);
    return;
  }
  const int k_groups = k / 4;
  const int blocks = n / block_n;
  const int tiles_per_block = block_n / 4;
  for (int block = 0; block < blocks; ++block) {
    for (int tile = 0; tile < tiles_per_block; ++tile) {
      int32x4_t dot = vdupq_n_s32(0);
      for (int group = 0; group < k_groups; ++group) {
        uint32_t x4;
        memcpy(&x4, x_q + group * 4, 4);
        const int8x16_t activation =
            vreinterpretq_s8_u32(vdupq_n_u32(x4));
        const int8_t *weight_ptr =
            packed + (((size_t)block * k_groups + group) * block_n
                      + tile * 4) * 4;
        dot = vdotq_s32(dot, activation, vld1q_s8(weight_ptr));
      }
      const int output_offset = block * block_n + tile * 4;
      float result[4];
      float32x4_t values = vmulq_n_f32(vcvtq_f32_s32(dot), scale);
      values = vmulq_f32(
          values, vld1q_f32(weight_scale + output_offset));
      vst1q_f32(result, values);
      for (int lane = 0; lane < 4; ++lane) {
        output[output_offset + lane] = float_to_bf16(result[lane]);
      }
    }
  }
}
