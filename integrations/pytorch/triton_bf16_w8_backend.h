#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct triton_bf16_w8_kernel triton_bf16_w8_kernel;
typedef struct triton_bf16_w8_mlp_kernel triton_bf16_w8_mlp_kernel;
typedef struct triton_bf16_rms_kernel triton_bf16_rms_kernel;
typedef struct triton_bf16_rope_kernel triton_bf16_rope_kernel;

triton_bf16_w8_kernel *triton_bf16_w8_kernel_create(
    const char *quant_kernel_dir, const char *gemv_kernel_dir, int64_t k,
    int64_t n);

triton_bf16_w8_kernel *triton_bf16_w8_kernel_create_wide(
    const char *quant_kernel_dir, const char *gemv_kernel_dir, int64_t k,
    int64_t n, int64_t block_n);

void triton_bf16_w8_kernel_destroy(triton_bf16_w8_kernel *kernel);

size_t triton_bf16_w8_packed_size(int64_t k, int64_t n);

int triton_bf16_w8_launch(triton_bf16_w8_kernel *kernel,
                          const uint16_t *x_bf16,
                          const int8_t *packed_weight,
                          const float *weight_scale,
                          uint16_t *output_bf16);

triton_bf16_w8_mlp_kernel *triton_bf16_w8_mlp_kernel_create(
    const char *quant_kernel_dir, const char *gemv_kernel_dir,
    const char *activation_kernel_dir, int64_t k, int64_t n,
    int64_t block_n);

void triton_bf16_w8_mlp_kernel_destroy(
    triton_bf16_w8_mlp_kernel *kernel);

int triton_bf16_w8_mlp_launch(triton_bf16_w8_mlp_kernel *kernel,
                              const uint16_t *x_bf16,
                              const int8_t *packed_gate_up,
                              const float *gate_up_scale,
                              uint16_t *output_bf16);

triton_bf16_rms_kernel *triton_bf16_rms_kernel_create(
    const char *kernel_dir, int64_t rows);

triton_bf16_rms_kernel *triton_bf16_fused_add_rms_kernel_create(
    const char *kernel_dir, int64_t rows);

void triton_bf16_rms_kernel_destroy(triton_bf16_rms_kernel *kernel);

int triton_bf16_rms_launch(triton_bf16_rms_kernel *kernel,
                           const uint16_t *x_bf16,
                           const uint16_t *weight_bf16,
                           uint16_t *output_bf16);

int triton_bf16_fused_add_rms_launch(triton_bf16_rms_kernel *kernel,
                                     uint16_t *input_bf16,
                                     uint16_t *residual_bf16,
                                     const uint16_t *weight_bf16);

triton_bf16_rope_kernel *triton_bf16_rope_kernel_create(
    const char *kernel_dir, int64_t total_heads);

void triton_bf16_rope_kernel_destroy(triton_bf16_rope_kernel *kernel);

int triton_bf16_rope_launch(triton_bf16_rope_kernel *kernel,
                            uint16_t *query_bf16, uint16_t *key_bf16,
                            const uint16_t *cosine_bf16,
                            const uint16_t *sine_bf16);

const char *triton_bf16_w8_last_error(void);

#ifdef __cplusplus
}
#endif
