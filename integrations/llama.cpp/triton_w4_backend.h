#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct triton_w4_kernel triton_w4_kernel;

triton_w4_kernel *triton_w4_kernel_create(
    const char *kernel_so, int64_t k, int64_t n);
triton_w4_kernel *triton_w4_kernel_create_from_bundle(
    const char *bundle_dir, int64_t k, int64_t n);
void triton_w4_kernel_destroy(triton_w4_kernel *kernel);

// Compiler layout:
//   packed bytes: [N/4, K/32, 4, 4, 4] = [tile, group, K4, N, K]
//   scales:       [N/4, K/32, 4] fp32
size_t triton_w4_packed_size(int64_t k, int64_t n);
size_t triton_w4_scale_count(int64_t k, int64_t n);

// Repack canonical row-major ggml block_q4_0 data. Each input block is
// exactly 18 bytes: fp16 d followed by 16 nibble bytes.
int triton_w4_pack_q4_0(const void *q4_blocks_nk, uint8_t *packed,
                        float *scales, int64_t k, int64_t n);

// Split canonical ggml block_q8_0 activation data. Each input block is
// exactly 34 bytes: fp16 d followed by 32 signed bytes.
int triton_w4_unpack_q8_0(const void *q8_blocks, int8_t *x_q,
                          float *x_scales, int64_t k);

int triton_w4_launch_q8(
    triton_w4_kernel *kernel, const int8_t *x_q, const float *x_scales,
    const uint8_t *packed, const float *weight_scales, float *output);

// Convenience path for ggml integration. Repacking and kernel loading are
// cached by the stable canonical weight pointer; activation scratch is
// thread-local.
int triton_w4_cached_q4_0_q8_0(
    const char *bundle_dir, const void *q4_blocks_nk,
    const void *q8_blocks, float *output, int64_t k, int64_t n);

// Extra-buffer integration: cache by the stable repacked tensor pointer while
// consuming canonical Q4_0 bytes supplied during ggml's model-load repack.
int triton_w4_cache_q4_0(
    const char *bundle_dir, const void *weight_key,
    const void *q4_blocks_nk, int64_t k, int64_t n);
int triton_w4_cached_launch_q8(
    const void *weight_key, const void *q8_blocks,
    float *output, int64_t k, int64_t n);
int triton_w4_cached_launch_q8_range(
    const void *weight_key, const void *q8_blocks,
    float *output, int64_t k, int64_t n,
    int64_t output_begin, int64_t output_end);
// Split prepare/launch API for a persistent CPU threadpool.  Preparation
// resolves the weight cache and unpacks the activation once into thread-local
// storage; any number of output ranges can then be launched without another
// mutex lookup or activation copy.
int triton_w4_cached_prepare_q8(
    const void *weight_key, const void *q8_blocks,
    int64_t k, int64_t n);
int triton_w4_prepared_launch_q8_range(
    float *output, int64_t output_begin, int64_t output_end);

// Q4_1 x Q8_1 uses the same dot-ready nibble layout plus the affine
// correction (m + 8*d) * (d_x * sum(q_x)).
int triton_w4_cache_q4_1(
    const char *bundle_dir, const void *weight_key,
    const void *q4_blocks_nk, int64_t k, int64_t n);
int triton_w4_cached_launch_q8_1(
    const void *weight_key, const void *q8_blocks,
    float *output, int64_t k, int64_t n);
int triton_w4_cached_launch_q8_1_range(
    const void *weight_key, const void *q8_blocks,
    float *output, int64_t k, int64_t n,
    int64_t output_begin, int64_t output_end);
int triton_w4_cached_prepare_q8_1(
    const void *weight_key, const void *q8_blocks,
    int64_t k, int64_t n);
int triton_w4_prepared_launch_q8_1_range(
    float *output, int64_t output_begin, int64_t output_end);

// Release every cached weight whose stable tensor key is inside a backend
// buffer that is about to be freed. This makes cache lifetime follow ggml's
// model buffers and prevents a later model from reusing stale packed weights
// at the same virtual address.
size_t triton_w4_cache_release_range(const void *base, size_t size);

const char *triton_w4_last_error(void);

#if defined(__cplusplus)
}
#endif
