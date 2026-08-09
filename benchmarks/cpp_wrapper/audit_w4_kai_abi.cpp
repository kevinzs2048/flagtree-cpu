#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"

// Include the production implementation so the internal model-load packer is
// tested byte-for-byte rather than duplicated in the audit program.
#include "../../integrations/llama.cpp/triton_w4_backend.cpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void store_fp16(uint8_t *destination, float value) {
    const __fp16 half = static_cast<__fp16>(value);
    std::memcpy(destination, &half, sizeof(half));
}

}  // namespace

int main() {
    constexpr size_t k = 96;
    constexpr size_t n = 12;
    constexpr size_t block = 32;
    constexpr size_t nr = 4;
    constexpr size_t kr = 16;
    constexpr size_t sr = 2;
    const size_t groups = k / block;

    std::vector<uint8_t> canonical_rhs(n * groups * 18);
    for (size_t row = 0; row < n; ++row) {
        for (size_t group = 0; group < groups; ++group) {
            uint8_t *source = canonical_rhs.data() +
                              (row * groups + group) * 18;
            store_fp16(source, 0.001f +
                                   static_cast<float>(row * groups + group) *
                                       0.0001f);
            for (size_t byte = 0; byte < 16; ++byte) {
                const uint8_t low =
                    static_cast<uint8_t>((row * 5 + group * 3 + byte) & 15);
                const uint8_t high = static_cast<uint8_t>(
                    (row * 7 + group * 11 + byte * 3) & 15);
                source[2 + byte] =
                    static_cast<uint8_t>(low | (high << 4));
            }
        }
    }

    const size_t official_rhs_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
            n, k, nr, kr, block);
    const size_t production_rhs_size = kai_packed_size(k, n);
    if (official_rhs_size != production_rhs_size) {
        throw std::runtime_error("RHS packed-size contract differs");
    }
    std::vector<uint8_t> official_rhs(official_rhs_size);
    std::vector<uint8_t> production_rhs(production_rhs_size);
    const kai_rhs_pack_qs4cxs1s0_param params{
        .lhs_zero_point = 1, .rhs_zero_point = 8};
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
        1, n, k, nr, kr, sr, block, canonical_rhs.data(), nullptr,
        official_rhs.data(), 0, &params);
    pack_q4_0_kai(
        canonical_rhs.data(), production_rhs.data(), k, n);
    if (official_rhs != production_rhs) {
        const auto mismatch = std::mismatch(
            official_rhs.begin(), official_rhs.end(), production_rhs.begin());
        throw std::runtime_error(
            "RHS byte mismatch at offset " +
            std::to_string(mismatch.first - official_rhs.begin()));
    }

    std::vector<float> lhs(k);
    std::vector<uint8_t> canonical_lhs(groups * 34);
    for (size_t group = 0; group < groups; ++group) {
        uint8_t *destination = canonical_lhs.data() + group * 34;
        store_fp16(destination, 0.5f);
        for (size_t lane = 0; lane < block; ++lane) {
            const int8_t quant = lane == 0
                                     ? 127
                                     : static_cast<int8_t>(
                                           (group * 31 + lane * 17) % 255 -
                                           127);
            destination[2 + lane] = static_cast<uint8_t>(quant);
            lhs[group * block + lane] = static_cast<float>(quant) * 0.5f;
        }
    }
    const size_t official_lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(
            1, k, block, 1, kr, sr);
    if (official_lhs_size != canonical_lhs.size()) {
        throw std::runtime_error("LHS packed-size contract differs");
    }
    std::vector<uint8_t> official_lhs(official_lhs_size);
    kai_run_lhs_quant_pack_qsi8d32p_f32(
        1, k, block, 1, kr, sr, 0, lhs.data(), k * sizeof(float),
        official_lhs.data());
    if (official_lhs != canonical_lhs) {
        const auto mismatch = std::mismatch(
            official_lhs.begin(), official_lhs.end(), canonical_lhs.begin());
        throw std::runtime_error(
            "LHS byte mismatch at offset " +
            std::to_string(mismatch.first - official_lhs.begin()));
    }

    std::cout << "PASS W4 KAI ABI audit rhs_bytes=" << official_rhs_size
              << " lhs_bytes=" << official_lhs_size
              << " rhs_exact=true lhs_exact=true\n";
    return 0;
}
