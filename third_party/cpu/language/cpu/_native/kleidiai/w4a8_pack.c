// SPDX-FileCopyrightText: Copyright 2026 BAAI
// SPDX-License-Identifier: Apache-2.0

/* Offline RHS packing (channelwise and blockwise) and optional profiling. */
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4cxp/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "w4a8_layout.h"

#if defined(__GNUC__)
#define FLAGTREE_KAI_EXPORT __attribute__((visibility("default")))
#else
#define FLAGTREE_KAI_EXPORT
#endif

#define UKERNEL matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod
#define _CAT(a, b) a##b
#define CAT(a, b) _CAT(a, b)
#define KGET(name) CAT(CAT(kai_get_, name), CAT(_, UKERNEL))

/* The grouped path packs against the dotprod ukernel's nr/kr/sr.  Both c32
 * ukernels must agree on those, otherwise one packed buffer could not serve
 * decode and prefill; kai_w4a8_selftest() asserts it at build time. */
#define UKERNEL_G matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod
#define KGET_G(name) CAT(CAT(kai_get_, name), CAT(_, UKERNEL_G))

FLAGTREE_KAI_EXPORT size_t flagtree_kai_w4a8_rhs_packed_size(
    size_t n, size_t k) {
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
        n, k, KGET(nr)(), KGET(kr)(), KGET(sr)());
}

FLAGTREE_KAI_EXPORT void flagtree_kai_w4a8_pack_rhs(
    size_t n, size_t k, const uint8_t *native_rhs,
    const float *scales, void *packed_rhs) {
    struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;

    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
        1, n, k, KGET(nr)(), KGET(kr)(), KGET(sr)(), native_rhs,
        NULL, scales, packed_rhs, 0, &params);
}

/* Blockwise (qsi4c32) packing.  Scales stay BF16 all the way to the ukernel:
 * that is the checkpoint's own dtype, so there is no conversion step and no
 * rounding introduced on top of what GPTQ already chose. */
FLAGTREE_KAI_EXPORT size_t flagtree_kai_w4a8_grouped_rhs_packed_size(
    size_t n, size_t k, size_t bl) {
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
        n, k, KGET_G(nr)(), KGET_G(kr)(), KGET_G(sr)(), bl, kai_dt_bf16);
}

FLAGTREE_KAI_EXPORT void flagtree_kai_w4a8_grouped_pack_rhs(
    size_t n, size_t k, size_t bl, const uint8_t *native_rhs,
    const void *scales, size_t scale_stride, void *packed_rhs) {
    struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = kai_dt_bf16;

    kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
        1, n, k, KGET_G(nr)(), KGET_G(kr)(), KGET_G(sr)(), bl, native_rhs,
        k / 2, NULL, scales, scale_stride, packed_rhs, 0, &params);
}

/* Reports the header size and validates the two c32 ukernels share a packed
 * RHS layout.  Returns 0 when they disagree so the Python side can refuse to
 * use the grouped path rather than silently computing wrong results. */
FLAGTREE_KAI_EXPORT size_t flagtree_kai_w4a8_header_bytes(void) {
    if (kai_get_nr_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm() !=
            KGET_G(nr)() ||
        kai_get_kr_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm() !=
            KGET_G(kr)() ||
        kai_get_sr_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm() !=
            KGET_G(sr)()) {
        return 0;
    }
    return FL_W4A8_HEADER_BYTES;
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
