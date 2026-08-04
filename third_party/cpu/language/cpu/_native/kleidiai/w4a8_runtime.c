// SPDX-FileCopyrightText: Copyright 2026 BAAI
// SPDX-License-Identifier: Apache-2.0

/* KleidiAI channelwise W4A8 backing function for FlagTree's CPU TLE op.
 *
 * The TLE ABI stays (x, packed_rhs, out, M, K, N).  M dispatch remains in C
 * so a single dynamic compiled graph can use dotprod for decode and i8mm for
 * prefill.  The parallel decomposition intentionally follows PyTorch's
 * KleidiAI channelwise implementation: pack LHS over M, then split matmul only
 * over N and let each ukernel call process the complete M stripe.
 */
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <float.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"

#if defined(__GNUC__)
#define FLAGTREE_KAI_EXPORT __attribute__((visibility("default")))
#else
#define FLAGTREE_KAI_EXPORT
#endif

typedef size_t (*get_value_fn)(void);
typedef size_t (*get_rhs_offset_fn)(size_t n_idx, size_t k);
typedef size_t (*get_dst_offset_fn)(
    size_t m_idx, size_t n_idx, size_t dst_stride);
typedef void (*run_matmul_fn)(
    size_t m, size_t n, size_t k, const void *lhs_packed,
    const void *rhs_packed, void *dst, size_t dst_stride_row,
    size_t dst_stride_col, float clamp_min, float clamp_max);

struct w4a8_kernel {
    get_value_fn get_n_step;
    get_value_fn get_mr;
    get_value_fn get_kr;
    get_value_fn get_sr;
    get_rhs_offset_fn get_rhs_offset;
    get_dst_offset_fn get_dst_offset;
    run_matmul_fn run;
};

static const struct w4a8_kernel gemv = {
    kai_get_n_step_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    kai_get_mr_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    kai_get_kr_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    kai_get_sr_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    kai_get_rhs_packed_offset_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    kai_get_dst_offset_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
    kai_run_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod,
};

static const struct w4a8_kernel gemm = {
    kai_get_n_step_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    kai_get_mr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    kai_get_kr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    kai_get_sr_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    kai_get_rhs_packed_offset_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    kai_get_dst_offset_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
    kai_run_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm,
};

void flagtree_kai_w4a8_profile_record(
    size_t m, size_t n, size_t k, uint64_t lhs_pack_ns,
    uint64_t compute_ns);

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

static size_t divide_round_up(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

static size_t round_up(size_t value, size_t step) {
    return divide_round_up(value, step) * step;
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

static void pack_lhs_chunk(
    size_t worker, size_t rows_per_worker, size_t worker_count,
    size_t m, size_t k, size_t mr, size_t kr, size_t sr,
    const uint16_t *x, void *lhs) {
    const size_t m_idx = worker * rows_per_worker;
    const size_t rows = worker + 1 == worker_count
        ? m - m_idx
        : rows_per_worker;
    uint8_t *lhs_chunk = (uint8_t *)lhs +
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(
            m_idx, k, mr, kr, sr);
    kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
        rows, k, mr, kr, sr, 0, x + m_idx * k,
        k * sizeof(uint16_t), lhs_chunk);
}

static void run_n_stripe(
    size_t worker, size_t cols_per_worker, size_t worker_count,
    size_t m, size_t n, size_t k, const struct w4a8_kernel *kernel,
    const void *lhs, const void *rhs, uint16_t *out) {
    const size_t n_idx = worker * cols_per_worker;
    const size_t cols = worker + 1 == worker_count
        ? n - n_idx
        : cols_per_worker;
    const uint8_t *rhs_chunk =
        (const uint8_t *)rhs + kernel->get_rhs_offset(n_idx, k);
    uint8_t *dst_chunk = (uint8_t *)out +
        kernel->get_dst_offset(0, n_idx, n * sizeof(uint16_t));
    kernel->run(
        m, cols, k, lhs, rhs_chunk, dst_chunk, n * sizeof(uint16_t),
        sizeof(uint16_t), -FLT_MAX, FLT_MAX);
}

FLAGTREE_KAI_EXPORT void flagtree_kai_w4a8_linear(
    const uint16_t *x, const void *rhs, uint16_t *out,
    int64_t m_value, int64_t k_value, int64_t n_value) {
    const size_t m = (size_t)m_value;
    const size_t k = (size_t)k_value;
    const size_t n = (size_t)n_value;
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    const struct w4a8_kernel *kernel = m == 1 ? &gemv : &gemm;
    const size_t mr = kernel->get_mr();
    const size_t kr = kernel->get_kr();
    const size_t sr = kernel->get_sr();
    size_t total_threads = (size_t)omp_get_max_threads();
    if (total_threads == 0) {
        total_threads = 1;
    }

    const int profile = w4a8_profile_enabled();
    const uint64_t start_ns = profile ? monotonic_ns() : 0;
    const size_t lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(
            m, k, mr, kr, sr);
    void *lhs = reserve_lhs_scratch(lhs_size);

    const size_t n_step = kernel->get_n_step();

    uint64_t packed_ns = 0;
    if (total_threads == 1) {
        pack_lhs_chunk(0, m, 1, m, k, mr, kr, sr, x, lhs);
    } else {
#pragma omp parallel
        {
            const size_t worker = (size_t)omp_get_thread_num();
            const size_t worker_count = (size_t)omp_get_num_threads();
            const size_t rows_per_worker = round_up(
                divide_round_up(m, worker_count), mr);
            const size_t pack_workers = divide_round_up(
                m, rows_per_worker);
            if (worker < pack_workers) {
                pack_lhs_chunk(
                    worker, rows_per_worker, pack_workers, m, k,
                    mr, kr, sr, x, lhs);
            }
        }
    }
    packed_ns = profile ? monotonic_ns() : 0;

    if (total_threads == 1) {
        run_n_stripe(0, n, 1, m, n, k, kernel, lhs, rhs, out);
    } else {
#pragma omp parallel
        {
            const size_t worker = (size_t)omp_get_thread_num();
            const size_t worker_count = (size_t)omp_get_num_threads();
            const size_t cols_per_worker = round_up(
                divide_round_up(n, worker_count), n_step);
            const size_t compute_workers = divide_round_up(
                n, cols_per_worker);
            if (worker < compute_workers) {
                run_n_stripe(
                    worker, cols_per_worker, compute_workers,
                    m, n, k, kernel, lhs, rhs, out);
            }
        }
    }

    if (profile) {
        const uint64_t end_ns = monotonic_ns();
        flagtree_kai_w4a8_profile_record(
            m, n, k, packed_ns - start_ns, end_ns - packed_ns);
    }
}
#endif
