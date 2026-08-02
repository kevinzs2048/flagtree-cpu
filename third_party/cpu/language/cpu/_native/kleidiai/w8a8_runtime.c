// SPDX-FileCopyrightText: Copyright 2026 BAAI
// SPDX-License-Identifier: Apache-2.0

/* ARM W8A8 linear via KleidiAI qai8dxp x qsi8cxp ukernels.
 *
 * Weights: per-row symmetric int8 (qsi8cxp4x8 packed once at load).
 * Activations: dynamic per-row asymmetric int8 (qai8dxp, packed per call).
 * Decode (m==1) uses the 1x4 NEON dotprod GEMV; prefill (m>1) uses the
 * 16x4 NEON i8mm GEMM.  Both share the same rhs packing (nr=4, kr=8, sr=1).
 * Kernels emit f32; we convert to bf16 in the same thread/tile for locality.
 */
#include <float.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"
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
    const size_t kr = VGET(kr)();
    const size_t sr = VGET(sr)();
    void *lhs = lhs_scratch(
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(1, k, mr, kr, sr));
    kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
        1, k, mr, kr, sr, 0, x, k * sizeof(uint16_t), lhs);

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
    const size_t kr = MGET(kr)();
    const size_t sr = MGET(sr)();
    const size_t nr = MGET(nr)();
    size_t n_step = MGET(n_step)();
    if (n_step < nr) n_step = nr;

    void *lhs = lhs_scratch(
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(m, k, mr, kr, sr));
    float *dst = (float *)dst_scratch(m * n * sizeof(float));
    const size_t m_tiles = (m + mr - 1) / mr;
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
                kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(
                    m0, k, mr, kr, sr);
            kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
                rows, k, mr, kr, sr, 0, x + m0 * k, k * sizeof(uint16_t), lhs_t);
        }
#pragma omp barrier

        const size_t n_per = (n_tiles + nthreads - 1) / nthreads;
        const size_t nt0 = tid * n_per;
        size_t nt1 = nt0 + n_per;
        if (nt1 > n_tiles) nt1 = n_tiles;
        for (size_t t = nt0; t < nt1; ++t) {
            const size_t n0 = t * n_step;
            const size_t w = n0 + n_step <= n ? n_step : n - n0;
            const uint8_t *rhs_t =
                (const uint8_t *)rhs + MGET(rhs_packed_offset)(n0, k);
            MRun(m, w, k, lhs, rhs_t, dst + n0, n * sizeof(float),
                 sizeof(float), -FLT_MAX, FLT_MAX);
            for (size_t r = 0; r < m; ++r) {
                f32_to_bf16_row(dst + r * n + n0, out + r * n + n0, w);
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
