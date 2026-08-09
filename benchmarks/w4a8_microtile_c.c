#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

// Hand-written ceiling for the compiler-oriented Q4 layout used by
// bench_w4a8_codegen.py.  This is deliberately a standalone benchmark
// function, not a Triton runtime implementation.
//
// packed: [N / 32, K / 32, 16, 32] u8
// scale:  [N / 32, K / 32, 32] f32
// x/qscale are grouped Q8_0-style values with one f32 scale per 32 values.
__attribute__((noinline, visibility("default")))
void w4a8_microtile_c(const int8_t *restrict x,
                      const float *restrict x_scale,
                      const uint8_t *restrict packed,
                      const float *restrict weight_scale,
                      float *restrict output, int k, int n) {
    const int groups = k / 32;
    const int tiles = n / 32;
    const uint8x16_t low_mask = vdupq_n_u8(0x0f);
    const int8x16_t zero_point = vdupq_n_s8(8);
    const uint8x16_t transpose4 = {
        0, 4, 8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7, 11, 15,
    };

    for (int tile = 0; tile < tiles; ++tile) {
        float32x4_t result[8];
        for (int nb = 0; nb < 8; ++nb) {
            result[nb] = vdupq_n_f32(0.0f);
        }

        for (int group = 0; group < groups; ++group) {
            int32x4_t dot[8];
            for (int nb = 0; nb < 8; ++nb) {
                dot[nb] = vdupq_n_s32(0);
            }
            const uint8_t *group_packed =
                packed + ((size_t)tile * groups + group) * 16 * 32;

            for (int kb = 0; kb < 16; kb += 4) {
                uint32_t xa_lo;
                uint32_t xa_hi;
                __builtin_memcpy(&xa_lo, x + group * 32 + kb, 4);
                __builtin_memcpy(&xa_hi, x + group * 32 + 16 + kb, 4);
                const int8x16_t a_lo =
                    vreinterpretq_s8_u32(vdupq_n_u32(xa_lo));
                const int8x16_t a_hi =
                    vreinterpretq_s8_u32(vdupq_n_u32(xa_hi));

                for (int nb = 0; nb < 8; ++nb) {
                    uint32x4_t rows = vdupq_n_u32(0);
                    for (int row = 0; row < 4; ++row) {
                        uint32_t bytes;
                        __builtin_memcpy(
                            &bytes,
                            group_packed + (kb + row) * 32 + nb * 4,
                            4);
                        rows = vsetq_lane_u32(bytes, rows, row);
                    }
                    const uint8x16_t transposed =
                        vqtbl1q_u8(vreinterpretq_u8_u32(rows), transpose4);
                    const int8x16_t weight_lo = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(transposed, low_mask)),
                        zero_point);
                    const int8x16_t weight_hi = vsubq_s8(
                        vreinterpretq_s8_u8(vshrq_n_u8(transposed, 4)),
                        zero_point);
                    dot[nb] = vdotq_s32(dot[nb], a_lo, weight_lo);
                    dot[nb] = vdotq_s32(dot[nb], a_hi, weight_hi);
                }
            }

            const float xs = x_scale[group];
            const float *scales =
                weight_scale + ((size_t)tile * groups + group) * 32;
            for (int nb = 0; nb < 8; ++nb) {
                const float32x4_t scale =
                    vmulq_n_f32(vld1q_f32(scales + nb * 4), xs);
                result[nb] = vfmaq_f32(
                    result[nb], vcvtq_f32_s32(dot[nb]), scale);
            }
        }

        for (int nb = 0; nb < 8; ++nb) {
            vst1q_f32(output + tile * 32 + nb * 4, result[nb]);
        }
    }
}
