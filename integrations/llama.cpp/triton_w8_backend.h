#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct triton_w8_kernel triton_w8_kernel;
typedef struct triton_w8_split_kernel triton_w8_split_kernel;

triton_w8_kernel * triton_w8_kernel_create(
    const char * kernel_dir, int64_t k, int64_t n, int64_t block_n);
triton_w8_kernel * triton_w8_kernel_create_from_bundle(
    const char * bundle_dir, int64_t k, int64_t n, int64_t block_n);
void triton_w8_kernel_destroy(triton_w8_kernel * kernel);

size_t triton_w8_packed_size(int64_t k, int64_t n);
int triton_w8_pack_i8_nk(const int8_t * weight_nk, int8_t * packed,
                         int64_t k, int64_t n, int64_t block_n);

int triton_w8_launch_f32(triton_w8_kernel * kernel, const float * x,
                         const int8_t * packed, const float * weight_scale,
                         float * output);

int triton_w8_launch_f32_range(triton_w8_kernel * kernel, const float * x,
                               const int8_t * packed,
                               const float * weight_scale, float * output,
                               int64_t block_start, int64_t block_count);

triton_w8_split_kernel * triton_w8_split_kernel_create(
    const char * quant_kernel_dir, const char * gemv_kernel_dir,
    int64_t k, int64_t n, int64_t block_n);
triton_w8_split_kernel * triton_w8_split_kernel_create_from_bundle(
    const char * bundle_dir, int64_t k, int64_t n, int64_t block_n);
void triton_w8_split_kernel_destroy(triton_w8_split_kernel * kernel);

int triton_w8_split_quantize_f32(
    triton_w8_split_kernel * kernel, const float * x);
int triton_w8_split_launch_prequant_f32_range(
    triton_w8_split_kernel * kernel, const int8_t * packed,
    const float * weight_scale, float * output,
    int64_t block_start, int64_t block_count);
int triton_w8_split_launch_f32(
    triton_w8_split_kernel * kernel, const float * x,
    const int8_t * packed, const float * weight_scale, float * output);

const char * triton_w8_last_error(void);

#if defined(__cplusplus)
}
#endif
