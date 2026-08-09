#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" size_t kai_reference_qsi8cxp_size(size_t n, size_t k) {
  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
      n, k, 4, 8, 1);
}

extern "C" void kai_reference_pack_qsi8cxp(size_t n, size_t k,
                                             const int8_t *weight,
                                             const float *scale,
                                             void *packed) {
  std::vector<float> bias(n, 0.0f);
  const kai_rhs_pack_qsi8cx_params params{.lhs_zero_point = 1,
                                           .scale_multiplier = 1.0f};
  kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
      1, n, k, 4, 8, 1, weight, bias.data(), scale, packed, 0, &params);
}
