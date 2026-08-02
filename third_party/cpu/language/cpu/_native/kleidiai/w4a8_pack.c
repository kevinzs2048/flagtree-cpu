// SPDX-FileCopyrightText: Copyright 2026 BAAI
// SPDX-License-Identifier: Apache-2.0

/* Offline RHS packing and optional profiling for FlagTree's W4A8 runtime. */
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"

#if defined(__GNUC__)
#define FLAGTREE_KAI_EXPORT __attribute__((visibility("default")))
#else
#define FLAGTREE_KAI_EXPORT
#endif

#define UKERNEL matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod
#define _CAT(a, b) a##b
#define CAT(a, b) _CAT(a, b)
#define KGET(name) CAT(CAT(kai_get_, name), CAT(_, UKERNEL))

FLAGTREE_KAI_EXPORT size_t flagtree_kai_w4a8_rhs_packed_size(
    size_t n, size_t k, size_t block_length) {
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
        n, k, KGET(nr)(), KGET(kr)(), KGET(sr)(), block_length,
        kai_dt_bf16);
}

FLAGTREE_KAI_EXPORT void flagtree_kai_w4a8_pack_rhs(
    size_t n, size_t k, size_t block_length, const uint8_t *native_rhs,
    const uint16_t *bf16_scales, void *packed_rhs) {
    struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = kai_dt_bf16;

    const size_t rhs_stride = (k + 1) / 2;
    const size_t scale_stride =
        ((k + block_length - 1) / block_length) * sizeof(uint16_t);
    kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
        1, n, k, KGET(nr)(), KGET(kr)(), KGET(sr)(), block_length,
        native_rhs, rhs_stride, NULL, bf16_scales, scale_stride,
        packed_rhs, 0, &params);
}

#define FL_PROFILE_SLOTS 64

struct flagtree_kai_w4a8_profile_entry {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t calls;
    uint64_t lhs_pack_ns;
    uint64_t gemv_ns;
};

static struct flagtree_kai_w4a8_profile_entry profile_entries[FL_PROFILE_SLOTS];
static atomic_flag profile_lock = ATOMIC_FLAG_INIT;

static void lock_profile(void) {
    while (atomic_flag_test_and_set_explicit(
        &profile_lock, memory_order_acquire)) {
    }
}

static void unlock_profile(void) {
    atomic_flag_clear_explicit(&profile_lock, memory_order_release);
}

void flagtree_kai_w4a8_profile_record(size_t m, size_t n, size_t k,
                                      uint64_t lhs_pack_ns,
                                      uint64_t compute_ns) {
    lock_profile();
    for (size_t index = 0; index < FL_PROFILE_SLOTS; ++index) {
        struct flagtree_kai_w4a8_profile_entry *entry = &profile_entries[index];
        if ((entry->m == m && entry->n == n && entry->k == k) ||
            entry->calls == 0) {
            entry->m = m;
            entry->n = n;
            entry->k = k;
            entry->calls++;
            entry->lhs_pack_ns += lhs_pack_ns;
            entry->gemv_ns += compute_ns;
            break;
        }
    }
    unlock_profile();
}

FLAGTREE_KAI_EXPORT void flagtree_kai_w4a8_profile_reset(void) {
    lock_profile();
    memset(profile_entries, 0, sizeof(profile_entries));
    unlock_profile();
}

FLAGTREE_KAI_EXPORT size_t flagtree_kai_w4a8_profile_count(void) {
    size_t count = 0;
    lock_profile();
    while (count < FL_PROFILE_SLOTS && profile_entries[count].calls != 0) {
        ++count;
    }
    unlock_profile();
    return count;
}

FLAGTREE_KAI_EXPORT int flagtree_kai_w4a8_profile_get(
    size_t index, uint64_t *values) {
    if (index >= FL_PROFILE_SLOTS || values == NULL) {
        return 0;
    }
    lock_profile();
    const struct flagtree_kai_w4a8_profile_entry entry = profile_entries[index];
    unlock_profile();
    if (entry.calls == 0) {
        return 0;
    }
    values[0] = entry.m;
    values[1] = entry.n;
    values[2] = entry.k;
    values[3] = entry.calls;
    values[4] = entry.lhs_pack_ns;
    values[5] = entry.gemv_ns;
    return 1;
}
