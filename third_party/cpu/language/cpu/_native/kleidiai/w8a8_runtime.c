// SPDX-FileCopyrightText: Copyright 2026 BAAI
// SPDX-License-Identifier: Apache-2.0

/* ARM W8A8 linear via KleidiAI qai8dxp x qsi8cxp ukernels.
 *
 * Weights: per-row symmetric int8 (qsi8cxp4x8 packed once at load).
 * Activations: dynamic per-row symmetric int8 (qai8dxp layout, zero point 0).
 * Decode (m==1) uses the 1x4 NEON dotprod GEMV; prefill (m>1) uses the
 * 16x4 NEON i8mm GEMM.  Both share the same rhs packing (nr=4, kr=8, sr=1).
 * Kernels emit f32; we convert to bf16 in the same thread/tile for locality.
 */
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_neon.h>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

#if defined(__GNUC__)
#define FLAGTREE_KAI_EXPORT __attribute__((visibility("default")))
#else
#define FLAGTREE_KAI_EXPORT
#endif

#define VGET(sym) kai_get_##sym##_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod
#define VRun kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod
#define MGET(sym) kai_get_##sym##_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm
#define MRun kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm

/* The KAI kernels consume this qai8dxp packed layout:
 *   interleaved int8 rows, int32 row offsets, float row scales.
 * KAI's BF16 packer is asymmetric. Standard W8A8 checkpoints require
 * dynamic per-token symmetric activations, so FlagTree owns this small
 * layout-compatible packer and writes zero offsets.
 */
static inline size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

static inline size_t lhs_packed_stride(size_t k, size_t mr) {
    return mr * (round_up(k, 32) + sizeof(int32_t) + sizeof(float));
}

static inline size_t lhs_packed_size(size_t m, size_t k, size_t mr) {
    return round_up(m, mr) / mr * lhs_packed_stride(k, mr);
}

static inline size_t lhs_packed_offset(size_t m_idx, size_t k, size_t mr) {
    return m_idx / mr * lhs_packed_stride(k, mr);
}

static inline float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline void load_bf16x8(const uint16_t *src, float32x4_t *lo,
                               float32x4_t *hi) {
    const uint16x8_t values = vld1q_u16(src);
    const uint16x8_t zero = vdupq_n_u16(0);
    *lo = vreinterpretq_f32_u16(vzip1q_u16(zero, values));
    *hi = vreinterpretq_f32_u16(vzip2q_u16(zero, values));
}

static float row_absmax(const uint16_t *src, size_t k) {
    float32x4_t maximum = vdupq_n_f32(0.0f);
    size_t index = 0;
    for (; index + 8 <= k; index += 8) {
        float32x4_t lo, hi;
        load_bf16x8(src + index, &lo, &hi);
        maximum = vmaxq_f32(maximum, vabsq_f32(lo));
        maximum = vmaxq_f32(maximum, vabsq_f32(hi));
    }
    float result = vmaxvq_f32(maximum);
    for (; index < k; ++index) {
        result = fmaxf(result, fabsf(bf16_to_f32(src[index])));
    }
    return result;
}

static inline void quantize_block8(const uint16_t *src, float inverse_scale,
                                   int8_t *dst) {
    float32x4_t lo, hi;
    load_bf16x8(src, &lo, &hi);
    int16x8_t quantized = vcombine_s16(
        vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(lo, inverse_scale))),
        vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(hi, inverse_scale))));
    quantized = vmaxq_s16(quantized, vdupq_n_s16(-127));
    quantized = vminq_s16(quantized, vdupq_n_s16(127));
    vst1_s8(dst, vqmovn_s16(quantized));
}

static void pack_lhs_symmetric(const uint16_t *src, size_t src_stride_elements,
                               size_t m, size_t k, size_t mr, void *packed) {
    const size_t k_internal = round_up(k, 32);
    const size_t stride = lhs_packed_stride(k, mr);

    for (size_t row_base = 0; row_base < m; row_base += mr) {
        const size_t rows = row_base + mr <= m ? mr : m - row_base;
        uint8_t *group = (uint8_t *)packed + row_base / mr * stride;
        float *scales = (float *)(group + mr * k_internal +
                                  mr * sizeof(int32_t));
        memset(group, 0, stride);

        for (size_t row = 0; row < rows; ++row) {
            const uint16_t *row_src =
                src + (row_base + row) * src_stride_elements;
            const float absolute_max = row_absmax(row_src, k);
            const float scale =
                absolute_max > 0.0f ? absolute_max / 127.0f : 1.0f;
            const float inverse_scale = 1.0f / scale;

            size_t column = 0;
            for (; column + 8 <= k; column += 8) {
                int8_t *block = (int8_t *)group + column * mr + row * 8;
                quantize_block8(row_src + column, inverse_scale, block);
            }
            if (column < k) {
                int8_t *block = (int8_t *)group + column * mr + row * 8;
                for (size_t tail = 0; column + tail < k; ++tail) {
                    long value = lrintf(bf16_to_f32(row_src[column + tail]) *
                                        inverse_scale);
                    if (value < -127) value = -127;
                    if (value > 127) value = 127;
                    block[tail] = (int8_t)value;
                }
            }

            /* offsets are already zero from memset */
            scales[row] = scale;
        }
    }
}

static void *grow(void **buf, size_t *cap, size_t need) {
    if (*cap < need) {
        void *next = aligned_alloc(64, (need + 63) & ~(size_t)63);
        if (next == NULL) {
            abort();
        }
        free(*buf);
        *buf = next;
        *cap = need;
    }
    return *buf;
}

static void *lhs_scratch(size_t need) {
    static _Thread_local void *buf = NULL;
    static _Thread_local size_t cap = 0;
    return grow(&buf, &cap, need);
}

static void *dst_scratch(size_t need) {
    static _Thread_local void *buf = NULL;
    static _Thread_local size_t cap = 0;
    return grow(&buf, &cap, need);
}

static inline void f32_to_bf16_row(const float *src, uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        memcpy(&bits, &src[i], sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u &&
            (bits & 0x007fffffu)) {
            dst[i] = (uint16_t)((bits >> 16) | 0x0040u);
            continue;
        }
        bits += 0x7FFFu + ((bits >> 16) & 1u); /* round to nearest even */
        dst[i] = (uint16_t)(bits >> 16);
    }
}

FLAGTREE_KAI_EXPORT size_t flagtree_kai_w8a8_rhs_packed_size(
    size_t n, size_t k) {
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        n, k, MGET(nr)(), MGET(kr)(), MGET(sr)());
}

FLAGTREE_KAI_EXPORT void flagtree_kai_w8a8_pack_rhs(
    size_t n, size_t k, const int8_t *rhs_nxk, const float *scale_n,
    void *rhs_packed) {
    struct kai_rhs_pack_qsi8cx_params params = {
        .lhs_zero_point = 1,
        .scale_multiplier = 1.0f,
    };
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, n, k, MGET(nr)(), MGET(kr)(), MGET(sr)(), rhs_nxk,
        /*bias=*/NULL, scale_n, rhs_packed, /*extra_bytes=*/0, &params);
}

static void run_gemv(const uint16_t *x, const void *rhs, uint16_t *out,
                     size_t n, size_t k) {
    const size_t mr = VGET(mr)();
    void *lhs = lhs_scratch(lhs_packed_size(1, k, mr));
    pack_lhs_symmetric(x, k, 1, k, mr, lhs);

    float *dst = (float *)dst_scratch(n * sizeof(float));
    const size_t nr = VGET(nr)();
    size_t n_step = VGET(n_step)();
    if (n_step < nr) n_step = nr;
    const size_t n_tiles = (n + n_step - 1) / n_step;
#pragma omp parallel
    {
        const size_t nthreads = (size_t)omp_get_num_threads();
        const size_t tid = (size_t)omp_get_thread_num();
        const size_t per = (n_tiles + nthreads - 1) / nthreads;
        const size_t begin = tid * per;
        size_t end = begin + per;
        if (end > n_tiles) end = n_tiles;
        for (size_t t = begin; t < end; ++t) {
            const size_t n0 = t * n_step;
            const size_t w = n0 + n_step <= n ? n_step : n - n0;
            const uint8_t *rhs_t =
                (const uint8_t *)rhs + VGET(rhs_packed_offset)(n0, k);
            VRun(1, w, k, lhs, rhs_t, dst + n0, n * sizeof(float),
                 sizeof(float), -FLT_MAX, FLT_MAX);
            f32_to_bf16_row(dst + n0, out + n0, w);
        }
    }
}

static void run_gemm(const uint16_t *x, const void *rhs, uint16_t *out,
                     size_t m, size_t n, size_t k) {
    const size_t mr = MGET(mr)();
    const size_t nr = MGET(nr)();
    const size_t m_step = MGET(m_step)();
    size_t n_step = MGET(n_step)();
    if (n_step < nr) n_step = nr;

    void *lhs = lhs_scratch(lhs_packed_size(m, k, mr));
    float *dst = (float *)dst_scratch(m * n * sizeof(float));
    const size_t m_tiles = (m + mr - 1) / mr;
    /* Distribute large prefill work on a 2-D MxN grid instead of assigning
     * every M row for an N stripe to one thread.  The KAI ukernel is 16x4;
     * 32x48 work tiles keep several ukernel calls cache-local while exposing
     * enough independent work to the OpenMP team. */
    const size_t m_block = 2 * m_step;
    const size_t m_work = (m + m_block - 1) / m_block;
    const size_t n_block = 12 * n_step;
    const size_t n_work = (n + n_block - 1) / n_block;
    const size_t work = m_work * n_work;
    const size_t n_tiles = (n + n_step - 1) / n_step;

#pragma omp parallel
    {
        const size_t nthreads = (size_t)omp_get_num_threads();
        const size_t tid = (size_t)omp_get_thread_num();

        /* parallel lhs quant+pack over row blocks */
        const size_t m_per = (m_tiles + nthreads - 1) / nthreads;
        const size_t mt0 = tid * m_per;
        size_t mt1 = mt0 + m_per;
        if (mt1 > m_tiles) mt1 = m_tiles;
        const size_t m0 = mt0 * mr;
        if (m0 < m) {
            const size_t rows = mt1 * mr <= m ? (mt1 - mt0) * mr : m - m0;
            uint8_t *lhs_t = (uint8_t *)lhs +
                lhs_packed_offset(m0, k, mr);
            pack_lhs_symmetric(x + m0 * k, k, rows, k, mr, lhs_t);
        }
#pragma omp barrier

        if (m <= 128) {
            /* Decode and small continuous-batching shapes need coarse N
             * stripes.  A fine 2-D grid makes their tiny M tiles dominated by
             * ukernel-call and scheduling overhead. */
            const size_t n_per = (n_tiles + nthreads - 1) / nthreads;
            const size_t begin = tid * n_per;
            size_t end = begin + n_per;
            if (end > n_tiles) end = n_tiles;
            for (size_t tile = begin; tile < end; ++tile) {
                const size_t n0 = tile * n_step;
                const size_t w = n0 + n_step <= n ? n_step : n - n0;
                const uint8_t *rhs_t =
                    (const uint8_t *)rhs + MGET(rhs_packed_offset)(n0, k);
                MRun(m, w, k, lhs, rhs_t, dst + n0, n * sizeof(float),
                     sizeof(float), -FLT_MAX, FLT_MAX);
                for (size_t r = 0; r < m; ++r) {
                    f32_to_bf16_row(dst + r * n + n0,
                                    out + r * n + n0, w);
                }
            }
        } else {
            const size_t per = (work + nthreads - 1) / nthreads;
            const size_t begin = tid * per;
            size_t end = begin + per;
            if (end > work) end = work;
            for (size_t task = begin; task < end; ++task) {
                const size_t n_tile = task / m_work;
                const size_t m_tile = task % m_work;
                const size_t m0 = m_tile * m_block;
                const size_t n0 = n_tile * n_block;
                const size_t h = m0 + m_block <= m ? m_block : m - m0;
                const size_t w = n0 + n_block <= n ? n_block : n - n0;
                const uint8_t *lhs_t = (const uint8_t *)lhs +
                    lhs_packed_offset(m0, k, mr);
                const uint8_t *rhs_t =
                    (const uint8_t *)rhs + MGET(rhs_packed_offset)(n0, k);
                MRun(h, w, k, lhs_t, rhs_t, dst + m0 * n + n0,
                     n * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
                for (size_t r = 0; r < h; ++r) {
                    f32_to_bf16_row(dst + (m0 + r) * n + n0,
                                    out + (m0 + r) * n + n0, w);
                }
            }
        }
    }
}

FLAGTREE_KAI_EXPORT void flagtree_kai_w8a8_linear(
    const uint16_t *x_bf16, const void *rhs_packed, uint16_t *out_bf16,
    int64_t m_value, int64_t k_value, int64_t n_value) {
    const size_t m = (size_t)m_value;
    const size_t k = (size_t)k_value;
    const size_t n = (size_t)n_value;
    if (m == 1) {
        run_gemv(x_bf16, rhs_packed, out_bf16, n, k);
    } else {
        run_gemm(x_bf16, rhs_packed, out_bf16, m, n, k);
    }
}
