#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

// Independent C/ACLE ceiling for bench_w8a8_codegen.py.  Packed weights use
// [N / 4, K / 4, 4, 4], with each contiguous 16-byte tile already arranged
// as four SDOT output lanes by four K lanes.
__attribute__((noinline, visibility("default")))
void w8a8_microtile4_c(const int8_t *restrict x,
                       const float *restrict x_scale,
                       const int8_t *restrict packed,
                       const float *restrict weight_scale,
                       float *restrict output, int k, int n) {
  const int k_groups = k / 4;
  const int tiles = n / 4;
  for (int tile = 0; tile < tiles; ++tile) {
    int32x4_t dot = vdupq_n_s32(0);
    const int8_t *tile_packed = packed + (size_t)tile * k_groups * 16;
    for (int group = 0; group < k_groups; ++group) {
      uint32_t x4;
      __builtin_memcpy(&x4, x + group * 4, 4);
      const int8x16_t activation =
          vreinterpretq_s8_u32(vdupq_n_u32(x4));
      const int8x16_t weight =
          vld1q_s8(tile_packed + (size_t)group * 16);
      dot = vdotq_s32(dot, activation, weight);
    }
    float32x4_t result = vmulq_f32(
        vcvtq_f32_s32(dot), vld1q_f32(weight_scale + tile * 4));
    result = vmulq_n_f32(result, x_scale[0]);
    vst1q_f32(output + tile * 4, result);
  }
}
