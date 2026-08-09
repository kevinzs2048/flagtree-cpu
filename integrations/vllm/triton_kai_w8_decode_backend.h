#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct triton_kai_w8_decode_kernel triton_kai_w8_decode_kernel;

uint32_t triton_kai_w8_backend_abi_version(void);

triton_kai_w8_decode_kernel *triton_kai_w8_decode_kernel_create(
    const char *bundle_dir, int64_t k, int64_t n);
void triton_kai_w8_decode_kernel_destroy(
    triton_kai_w8_decode_kernel *kernel);

int triton_kai_w8_decode_launch(triton_kai_w8_decode_kernel *kernel,
                                const uint16_t *x_bf16,
                                const uint8_t *rhs_packed,
                                uint16_t *output_bf16);

int triton_kai_w8_prefill_launch(triton_kai_w8_decode_kernel *kernel,
                                 int64_t m, const uint16_t *x_bf16,
                                 const uint8_t *rhs_packed,
                                 uint16_t *output_bf16);

const char *triton_kai_w8_decode_last_error(void);

#if defined(__cplusplus)
}
#endif
