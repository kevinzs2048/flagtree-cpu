#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

// Independent C/ACLE ceiling for bench_w4a8_q41_codegen.py.  The packed
// weights use [N / 4, K / 32, 4, 4, 4], while all scale/offset arrays use
// [N / 4, K / 32, 4].  packed_offset stores Q4_1 (minimum + 8 * scale),
// matching the signed-nibble SDOT correction in the Triton kernel.
__attribute__((noinline, visibility("default")))
void w4a8_q41_microtile4_c(const int8_t *restrict x,
                           const float *restrict x_scale,
                           const float *restrict x_sum,
                           const uint8_t *restrict packed,
                           const float *restrict weight_scale,
                           const float *restrict packed_offset,
                           float *restrict output, int k, int n) {
  const int groups = k / 32;
  const int tiles = n / 4;
  const uint8x16_t low_mask = vdupq_n_u8(0x0f);
  const int8x16_t zero_point = vdupq_n_s8(8);

  for (int tile = 0; tile < tiles; ++tile) {
    float32x4_t result = vdupq_n_f32(0.0f);
    for (int group = 0; group < groups; ++group) {
      int32x4_t dot = vdupq_n_s32(0);
      const size_t scale_index = ((size_t)tile * groups + group) * 4;
      const uint8_t *group_packed = packed + scale_index * 16;

      for (int chunk = 0; chunk < 4; ++chunk) {
        uint32_t x_low;
        uint32_t x_high;
        __builtin_memcpy(&x_low, x + group * 32 + chunk * 4, 4);
        __builtin_memcpy(&x_high,
                         x + group * 32 + 16 + chunk * 4, 4);
        const int8x16_t a_low =
            vreinterpretq_s8_u32(vdupq_n_u32(x_low));
        const int8x16_t a_high =
            vreinterpretq_s8_u32(vdupq_n_u32(x_high));

        const uint8x16_t q =
            vld1q_u8(group_packed + (size_t)chunk * 16);
        const int8x16_t weight_low = vsubq_s8(
            vreinterpretq_s8_u8(vandq_u8(q, low_mask)), zero_point);
        const int8x16_t weight_high = vsubq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), zero_point);
        dot = vdotq_s32(dot, a_low, weight_low);
        dot = vdotq_s32(dot, a_high, weight_high);
      }

      const float32x4_t scale = vmulq_n_f32(
          vld1q_f32(weight_scale + scale_index), x_scale[group]);
      result = vfmaq_f32(result, vcvtq_f32_s32(dot), scale);
      result = vfmaq_n_f32(
          result, vld1q_f32(packed_offset + scale_index), x_sum[group]);
    }
    vst1q_f32(output + tile * 4, result);
  }
}
