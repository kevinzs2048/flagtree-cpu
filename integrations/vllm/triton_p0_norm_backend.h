#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct triton_p0_norm_kernel triton_p0_norm_kernel;
typedef struct triton_p0_rope_kernel triton_p0_rope_kernel;

uint32_t triton_p0_norm_backend_abi_version(void);

triton_p0_norm_kernel *triton_p0_norm_kernel_create(
    const char *kernel_dir, int32_t fused_add, int64_t rows);

void triton_p0_norm_kernel_destroy(triton_p0_norm_kernel *kernel);

int triton_p0_rms_launch(triton_p0_norm_kernel *kernel,
                         const uint16_t *input_bf16,
                         const uint16_t *weight_bf16,
                         uint16_t *output_bf16);

int triton_p0_fused_add_rms_launch(triton_p0_norm_kernel *kernel,
                                   uint16_t *input_bf16,
                                   uint16_t *residual_bf16,
                                   const uint16_t *weight_bf16);

const char *triton_p0_norm_last_error(void);

triton_p0_rope_kernel *triton_p0_rope_kernel_create(
    const char *kernel_dir, int64_t total_heads);

void triton_p0_rope_kernel_destroy(triton_p0_rope_kernel *kernel);

int triton_p0_rope_launch(triton_p0_rope_kernel *kernel,
                          uint16_t *query_bf16, uint16_t *key_bf16,
                          const int64_t *positions,
                          const uint16_t *cos_sin_cache_bf16);

#ifdef __cplusplus
}
#endif
