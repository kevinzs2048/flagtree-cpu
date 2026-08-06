// SPDX-FileCopyrightText: Copyright 2026 BAAI
// SPDX-License-Identifier: Apache-2.0

/* Layout descriptor prefixed to every packed W4A8 RHS buffer.
 *
 * The TLE op ABI is (x, packed_rhs, out, M, K, N) and the runtime symbol name
 * is fixed by the MLIR lowering, so the block length cannot travel as an
 * argument.  Carrying it in the buffer instead keeps the ABI untouched and
 * makes the layout a property of the weight: a single process can mix a
 * grouped body with a channelwise lm_head, which is exactly what the
 * G128 + INT4-lm_head checkpoint needs.
 *
 * bl == 0 selects the channelwise (qsi4cxp) kernels; bl > 0 selects the
 * blockwise (qsi4c32p) kernels with that block length.
 */
#ifndef FLAGTREE_KAI_W4A8_LAYOUT_H
#define FLAGTREE_KAI_W4A8_LAYOUT_H

#include <stdint.h>

/* Keeps the payload 64-byte aligned so the ukernels see the same alignment
 * they would get from the start of a fresh allocation. */
#define FL_W4A8_HEADER_BYTES 64
#define FL_W4A8_MAGIC 0x464C57344138ULL /* "FLW4A8" */

struct fl_w4a8_header {
    uint64_t magic;
    uint64_t bl;
};

#endif
