// SPDX-FileCopyrightText: Copyright 2026 BAAI
// SPDX-License-Identifier: Apache-2.0

/* KleidiAI W4A8 backing function for the FlagTree CPU TLE op.
 *
 * The TLE op passes (x, packed_rhs, out, M, K, N).  Keep M dispatch inside
 * the compiled function so vLLM's guard-dropping DYNAMO_TRACE_ONCE graph can
 * serve both prefill and decode without specializing a Python shape branch.
 */
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <omp.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__GNUC__)
#define FLAGTREE_KAI_EXPORT __attribute__((visibility("default")))
#else
#define FLAGTREE_KAI_EXPORT
#endif

#define BL 32
#define _CAT(a, b) a##b
#define CAT(a, b) _CAT(a, b)

#define GEMV matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod
/* The f32 16x4x32 GEMM is measurably faster than the bf16 16x4 one (+4.8% on
 * end-to-end prefill), at the cost of emitting f32; the tile is narrowed to
 * bf16 here while it is still in cache.  Both share nr=4/kr=16/sr=2 with the
 * decode GEMV, so the RHS stays packed once for all paths. */
#define GEMM matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm
#define VGET(f) CAT(CAT(kai_get_, f), CAT(_, GEMV))
#define VRun CAT(kai_run_, GEMV)
#define MGET(f) CAT(CAT(kai_get_, f), CAT(_, GEMM))
#define MRun CAT(kai_run_, GEMM)

size_t VGET(mr)(void);
size_t VGET(kr)(void);
size_t VGET(sr)(void);
size_t VGET(nr)(void);
size_t VGET(n_step)(void);
size_t VGET(rhs_packed_offset)(size_t n_idx, size_t k, size_t bl);
size_t VGET(dst_offset)(size_t m_idx, size_t n_idx, size_t dst_stride);
void VRun(size_t m, size_t n, size_t k, size_t bl,
          const void *lhs_packed, const void *rhs_packed, void *dst,
          size_t dst_stride_row, size_t dst_stride_col,
          float clamp_min, float clamp_max);

size_t MGET(mr)(void);
size_t MGET(kr)(void);
size_t MGET(sr)(void);
size_t MGET(nr)(void);
size_t MGET(n_step)(void);
size_t MGET(rhs_packed_offset)(size_t n_idx, size_t k, size_t bl);
size_t MGET(dst_offset)(size_t m_idx, size_t n_idx, size_t dst_stride);
void MRun(size_t m, size_t n, size_t k, size_t bl,
          const void *lhs_packed, const void *rhs_packed, void *dst,
          size_t dst_stride_row, size_t dst_stride_col,
          float clamp_min, float clamp_max);

size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr);
size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);
void kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr,
    size_t m_idx_start, const void *lhs, size_t lhs_stride,
    void *lhs_packed);

void flagtree_kai_w4a8_profile_record(size_t m, size_t n, size_t k,
                                      uint64_t lhs_pack_ns,
                                      uint64_t compute_ns);

/* Per-thread f32 landing zone for one output tile of the GEMM, converted to
 * bf16 before moving on so the data is still in cache. */
static _Thread_local float *dst_scratch;
static _Thread_local size_t dst_scratch_capacity;

static float *reserve_dst_scratch(size_t floats) {
    if (floats <= dst_scratch_capacity) {
        return dst_scratch;
    }
    void *next = realloc(dst_scratch, floats * sizeof(float));
    if (next == NULL) {
        abort();
    }
    dst_scratch = (float *)next;
    dst_scratch_capacity = floats;
    return dst_scratch;
}

/* f32 -> bf16, round-to-nearest-even (matches torch's .to(bfloat16)). */
static inline uint16_t fp32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu)) {
        return (uint16_t)((bits >> 16) | 0x0040u);  /* keep NaN quiet */
    }
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

static _Thread_local uint8_t *lhs_scratch;
static _Thread_local size_t lhs_scratch_capacity;

static void *reserve_lhs_scratch(size_t size) {
    if (size <= lhs_scratch_capacity) {
        return lhs_scratch;
    }
    void *next = realloc(lhs_scratch, size);
    if (next == NULL) {
        abort();
    }
    lhs_scratch = (uint8_t *)next;
    lhs_scratch_capacity = size;
    return next;
}

static int w4a8_profile_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("FL_W4A8_PROFILE");
        enabled = value != NULL && strcmp(value, "1") == 0;
    }
    return enabled;
}

static uint64_t monotonic_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}

static void run_gemv(const uint16_t *x, const void *rhs, uint16_t *out,
                     size_t n, size_t k) {
    const int profile = w4a8_profile_enabled();
    const uint64_t start_ns = profile ? monotonic_ns() : 0;
    const size_t mr = VGET(mr)();
    const size_t kr = VGET(kr)();
    const size_t sr = VGET(sr)();
    const size_t lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(
            1, k, mr, kr, sr);
    void *lhs = reserve_lhs_scratch(lhs_size);
    kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
        1, k, mr, kr, sr, 0, x, k * sizeof(uint16_t), lhs);
    const uint64_t packed_ns = profile ? monotonic_ns() : 0;

    const size_t nr = VGET(nr)();
    size_t n_step = VGET(n_step)();
    if (n_step < nr) {
        n_step = nr;
    }
    const size_t n_tiles = (n + n_step - 1) / n_step;
#pragma omp parallel
    {
        const size_t thread_count = (size_t)omp_get_num_threads();
        const size_t thread_id = (size_t)omp_get_thread_num();
        const size_t tiles_per_thread =
            (n_tiles + thread_count - 1) / thread_count;
        const size_t begin = thread_id * tiles_per_thread;
        size_t end = begin + tiles_per_thread;
        if (end > n_tiles) {
            end = n_tiles;
        }
        for (size_t tile = begin; tile < end; ++tile) {
            const size_t n0 = tile * n_step;
            const size_t width = n0 + n_step <= n ? n_step : n - n0;
            const uint8_t *rhs_tile =
                (const uint8_t *)rhs + VGET(rhs_packed_offset)(n0, k, BL);
            uint16_t *out_tile = (uint16_t *)(
                (uint8_t *)out + VGET(dst_offset)(0, n0, n * sizeof(uint16_t)));
            VRun(1, width, k, BL, lhs, rhs_tile, out_tile,
                 n * sizeof(uint16_t), sizeof(uint16_t), -3.0e38f, 3.0e38f);
        }
    }
    if (profile) {
        const uint64_t end_ns = monotonic_ns();
        flagtree_kai_w4a8_profile_record(
            1, n, k, packed_ns - start_ns, end_ns - packed_ns);
    }
}

static void run_gemm(const uint16_t *x, const void *rhs, uint16_t *out,
                     size_t m, size_t n, size_t k) {
    const int profile = w4a8_profile_enabled();
    const uint64_t start_ns = profile ? monotonic_ns() : 0;
    uint64_t packed_ns = 0;
    const size_t mr = MGET(mr)();
    const size_t kr = MGET(kr)();
    const size_t sr = MGET(sr)();
    const size_t nr = MGET(nr)();
    size_t n_step = MGET(n_step)();
    if (n_step < nr) {
        n_step = nr;
    }

    const size_t lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(
            m, k, mr, kr, sr);
    void *lhs = reserve_lhs_scratch(lhs_size);
    const size_t m_tiles = (m + mr - 1) / mr;
    const size_t n_tiles = (n + n_step - 1) / n_step;

#pragma omp parallel
    {
        const size_t thread_id = (size_t)omp_get_thread_num();

        const size_t thread_count = (size_t)omp_get_num_threads();
        const size_t m_tiles_per_thread =
            (m_tiles + thread_count - 1) / thread_count;
        const size_t m_tile_begin = thread_id * m_tiles_per_thread;
        size_t m_tile_end = m_tile_begin + m_tiles_per_thread;
        if (m_tile_end > m_tiles) {
            m_tile_end = m_tiles;
        }
        const size_t m0 = m_tile_begin * mr;
        if (m0 < m) {
            const size_t rows = m_tile_end * mr <= m
                ? (m_tile_end - m_tile_begin) * mr
                : m - m0;
            uint8_t *lhs_tile = (uint8_t *)lhs +
                kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(
                    m0, k, mr, kr, sr);
            kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
                rows, k, mr, kr, sr, 0, x + m0 * k,
                k * sizeof(uint16_t), lhs_tile);
        }

#pragma omp barrier
        if (profile && thread_id == 0) {
            packed_ns = monotonic_ns();
        }
        if (profile) {
#pragma omp barrier
        }
        const size_t n_tiles_per_thread =
            (n_tiles + thread_count - 1) / thread_count;
        const size_t n_tile_begin = thread_id * n_tiles_per_thread;
        size_t n_tile_end = n_tile_begin + n_tiles_per_thread;
        if (n_tile_end > n_tiles) {
            n_tile_end = n_tiles;
        }
        float *tile_f32 = reserve_dst_scratch(m * n_step);
        for (size_t tile = n_tile_begin; tile < n_tile_end; ++tile) {
            const size_t n0 = tile * n_step;
            const size_t width = n0 + n_step <= n ? n_step : n - n0;
            const uint8_t *rhs_tile =
                (const uint8_t *)rhs + MGET(rhs_packed_offset)(n0, k, BL);
            /* The ukernel emits f32 into the thread-local tile ... */
            MRun(m, width, k, BL, lhs, rhs_tile, tile_f32,
                 width * sizeof(float), sizeof(float), -3.0e38f, 3.0e38f);
            /* ... and it is narrowed to bf16 straight into the output. */
            for (size_t row = 0; row < m; ++row) {
                const float *src = tile_f32 + row * width;
                uint16_t *dst = out + row * n + n0;
                for (size_t col = 0; col < width; ++col) {
                    dst[col] = fp32_to_bf16(src[col]);
                }
            }
        }
    }
    if (profile) {
        const uint64_t end_ns = monotonic_ns();
        flagtree_kai_w4a8_profile_record(
            m, n, k, packed_ns - start_ns, end_ns - packed_ns);
    }
}

FLAGTREE_KAI_EXPORT void flagtree_kai_w4a8_linear(
    const uint16_t *x, const void *rhs, uint16_t *out,
    int64_t m_value, int64_t k_value, int64_t n_value) {
    const size_t m = (size_t)m_value;
    const size_t k = (size_t)k_value;
    const size_t n = (size_t)n_value;
    if (m == 1) {
        run_gemv(x, rhs, out, n, k);
    } else {
        run_gemm(x, rhs, out, m, n, k);
    }
}
#endif
