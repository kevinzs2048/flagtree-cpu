/*
 * SME prefill attention (M>1) — fully fused QK^T -> softmax -> P*V, bf16 lossless.
 *
 * Companion to runtime_flash_attn_decode.cpp (M=1 NEON path). This is the M>1
 * prefill path: the two score matmuls run on the SME unit via BFMOPA
 * (sme_tile_16x64_bf16), the row-wise softmax runs in scalar/NEON.
 *
 * PROVENANCE / CONFIDENCE:
 *   - QK^T packing + tiling is a faithful port of the VERIFIED llama.cpp impl
 *     ggml-cpu/flaggems/flaggems.cpp::aux_op_traits::forward_attn_qkt
 *     (numerically validated on M4 Pro, 2026-06-26). HIGH confidence.
 *   - Row-wise softmax + causal mask: straightforward. HIGH confidence.
 *   - P*V (symmetric second GEMM) is NEW here (in llama.cpp P*V was a second
 *     ggml MUL_MAT handled by the same kernel; here it is inlined). The packing
 *     math mirrors QK^T by construction but has NOT been compiled/tested because
 *     this machine's FlagTree toolchain is currently broken (missing tle-m4 +
 *     pinned LLVM). MUST run test_sme_attn_prefill.py once the toolchain builds.
 *
 * Drop this file into FlagTree/third_party/cpu/runtime/ and add it to that
 * directory's CMakeLists.txt TritonCPURuntime sources (see README).
 */

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <omp.h>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

// SME bf16 tile kernel (runtime_sme_kernel.S). C[16][..] += Ap @ Bp^T over
// K2*2.  Keep the shared library loadable when the build assembler has no SME2
// support: Linux linkers permit unresolved symbols in DSOs, so an unconditional
// declaration here otherwise produces a runtime dlopen failure even though
// tle_cpu_has_sme() correctly prevents execution of this path.
#ifndef TLE_NO_SME_KERNEL
extern "C" void sme_tile_16x64_bf16(const uint16_t *Ap, const uint16_t *Bp,
                                    float *C, int64_t K2, int64_t ldc_elems);
#else
static void sme_tile_16x64_bf16(const uint16_t *, const uint16_t *, float *,
                                int64_t, int64_t) {
  fprintf(stderr,
          "TritonCPURuntime: sme_tile_16x64_bf16 called but the runtime "
          "was built without the SME kernel\n");
  abort();
}
#endif
// SME availability gate (cpu_features.cpp). Callers must consult before dispatch;
// we also fail loudly here as defense-in-depth (mirrors sme_gemm_bf16).
extern "C" int tle_cpu_has_sme(void);

extern "C" {

#if defined(__aarch64__) && defined(__ARM_NEON)

// f32 -> bf16 round-to-nearest-even (matches tle_f2bf in runtime_sdot.cpp).
static inline uint16_t tle_f2bf(float f) {
  uint32_t x;
  __builtin_memcpy(&x, &f, 4);
  uint32_t r = x + 0x7FFFu + ((x >> 16) & 1u);
  return (uint16_t)(r >> 16);
}

// bf16 -> f32 (zero-extend mantissa).
static inline float tle_bf2f(uint16_t b) {
  uint32_t x = (uint32_t)b << 16;
  float f;
  __builtin_memcpy(&f, &x, 4);
  return f;
}

/*
 * sme_attn_prefill_bf16:
 *   Q:   [num_heads,    M,   head_dim] bf16 (contiguous)
 *   K:   [num_kv_heads, Nkv, head_dim] bf16 (contiguous)
 *   V:   [num_kv_heads, Nkv, head_dim] bf16 (contiguous)
 *   Out: [num_heads,    M,   head_dim] bf16 (contiguous)
 *   sm_scale: usually head_dim^-0.5
 *   causal:   1 => key n attends to query m iff n <= (Nkv-M)+m
 *
 * GQA via num_kv_heads < num_heads (broadcast ratio = num_heads/num_kv_heads).
 * Parallelized across query heads (disjoint Out writes, no barrier). The single
 * SME unit serializes BFMOPA; the per-head bf16 packing parallelizes across cores.
 */
EXPORT void sme_attn_prefill_bf16(const uint16_t *Q, const uint16_t *K,
                                  const uint16_t *V, uint16_t *Out, int64_t M,
                                  int64_t Nkv, int64_t head_dim, float sm_scale,
                                  int64_t num_heads, int64_t num_kv_heads,
                                  int64_t causal) {
  if (!tle_cpu_has_sme()) {
    fprintf(stderr, "TritonCPURuntime: sme_attn_prefill_bf16 called without "
                    "usable SME (check tle_cpu_has_sme() first)\n");
    abort();
  }

  const int64_t D = head_dim;
  const int64_t K2 = D / 2;                  // QK^T reduction (= D) in pairs
  const int64_t Npk = (Nkv + 63) / 64 * 64;  // QK^T out cols (Nkv padded to 64)
  const int64_t NkvE = (Nkv + 1) & ~1ll;     // Nkv padded to even (P*V reduction)
  const int64_t K2v = NkvE / 2;              // P*V reduction in pairs
  const int64_t Npv = (D + 63) / 64 * 64;    // P*V out cols (D padded to 64)
  const int64_t ratio = (num_kv_heads > 0) ? num_heads / num_kv_heads : 1;
  const int64_t kv_off = Nkv - M;            // past-context length (>=0 prefill)

#pragma omp parallel
  {
    // Per-thread scratch (heap; prefill panels are large).
    uint16_t *Bk = (uint16_t *)malloc((size_t)(Npk / 64) * K2 * 128 * sizeof(uint16_t));
    uint16_t *Ak = (uint16_t *)malloc((size_t)K2 * 32 * sizeof(uint16_t));
    float *S = (float *)malloc((size_t)16 * Npk * sizeof(float));
    uint16_t *P = (uint16_t *)malloc((size_t)16 * NkvE * sizeof(uint16_t)); // bf16 probs
    uint16_t *Av = (uint16_t *)malloc((size_t)K2v * 32 * sizeof(uint16_t));
    uint16_t *Bv = (uint16_t *)malloc((size_t)(Npv / 64) * K2v * 128 * sizeof(uint16_t));
    float *O = (float *)malloc((size_t)16 * Npv * sizeof(float));

#pragma omp for schedule(static)
    for (int64_t h = 0; h < num_heads; h++) {
      const int64_t hkv = h / ratio;
      const uint16_t *Qh = Q + h * M * D;
      const uint16_t *Kh = K + hkv * Nkv * D;
      const uint16_t *Vh = V + hkv * Nkv * D;
      uint16_t *Oh = Out + h * M * D;

      // ---- Pack K once for this head: Bk[nt][k2][b][j][p] = K[2k2+p, n]. ----
      // (verbatim from forward_attn_qkt; K rows here are bf16 not f16)
      std::memset(Bk, 0, (size_t)(Npk / 64) * K2 * 128 * sizeof(uint16_t));
      for (int64_t n = 0; n < Nkv; n++) {
        uint16_t *base = Bk + (n / 64) * K2 * 128 + ((n % 64) / 16) * 32 + (n % 16) * 2;
        const uint16_t *krow = Kh + n * D;
        for (int64_t k2 = 0; k2 < K2; k2++) {
          base[k2 * 128 + 0] = krow[2 * k2 + 0];
          base[k2 * 128 + 1] = krow[2 * k2 + 1];
        }
      }

      // ---- Strip over M in blocks of 16 rows. ----
      for (int64_t m0 = 0; m0 < M; m0 += 16) {
        const int64_t rows = (m0 + 16 <= M) ? 16 : (M - m0);

        // Pack Q strip: Ak[i][k2][p] = Q[2k2+p, m0+i].
        std::memset(Ak, 0, (size_t)K2 * 32 * sizeof(uint16_t));
        for (int64_t i = 0; i < rows; i++) {
          uint16_t *base = Ak + i * 2;
          const uint16_t *qrow = Qh + (m0 + i) * D;
          for (int64_t k2 = 0; k2 < K2; k2++) {
            base[k2 * 32 + 0] = qrow[2 * k2 + 0];
            base[k2 * 32 + 1] = qrow[2 * k2 + 1];
          }
        }

        // QK^T: compute only non-future 64-col tiles when causal.
        const int64_t ntiles = Npk / 64;
        int64_t comp_tiles = ntiles;
        if (causal) {
          const int64_t last_n = kv_off + m0 + rows - 1;
          const int64_t t = last_n / 64 + 1;
          comp_tiles = t < ntiles ? t : ntiles;
        }
        for (int64_t nt = 0; nt < comp_tiles; nt++) {
          sme_tile_16x64_bf16(Ak, Bk + nt * K2 * 128, S + nt * 64, K2, Npk);
        }
        const int64_t comp_cols = comp_tiles * 64;

        // ---- Row-wise softmax over [0,Nkv) with scale + causal mask -> P (bf16). ----
        std::memset(P, 0, (size_t)16 * NkvE * sizeof(uint16_t));
        for (int64_t i = 0; i < rows; i++) {
          const int64_t valid_n = causal
              ? (kv_off + m0 + i + 1 < Nkv ? kv_off + m0 + i + 1 : Nkv)
              : Nkv;
          const int64_t lim = valid_n < comp_cols ? valid_n : comp_cols;
          const float *srow = S + i * Npk;
          // max
          float mx = -1e30f;
          for (int64_t n = 0; n < lim; n++) {
            float s = srow[n] * sm_scale;
            if (s > mx) mx = s;
          }
          // exp + sum
          float sum = 0.0f;
          uint16_t *prow = P + i * NkvE;
          for (int64_t n = 0; n < lim; n++) {
            float e = expf(srow[n] * sm_scale - mx);
            sum += e;
            prow[n] = tle_f2bf(e);  // store unnormalized; fold 1/sum into V below
          }
          // normalize (fold into bf16 probs so P*V is a plain matmul)
          float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
          for (int64_t n = 0; n < lim; n++) {
            prow[n] = tle_f2bf(tle_bf2f(prow[n]) * inv);
          }
        }

        // ---- Pack P (activation) and V^T (weights) for P*V; reduction = Nkv. ----
        // Av[i][kv2][p] = P[i, 2kv2+p]
        std::memset(Av, 0, (size_t)K2v * 32 * sizeof(uint16_t));
        for (int64_t i = 0; i < rows; i++) {
          uint16_t *base = Av + i * 2;
          const uint16_t *prow = P + i * NkvE;
          for (int64_t kv2 = 0; kv2 < K2v; kv2++) {
            base[kv2 * 32 + 0] = prow[2 * kv2 + 0];
            base[kv2 * 32 + 1] = prow[2 * kv2 + 1];
          }
        }
        // Bv[dt][kv2][b][j][p] = V[2kv2+p, d]  (out col = d, reduction = n)
        std::memset(Bv, 0, (size_t)(Npv / 64) * K2v * 128 * sizeof(uint16_t));
        for (int64_t d = 0; d < D; d++) {
          uint16_t *base = Bv + (d / 64) * K2v * 128 + ((d % 64) / 16) * 32 + (d % 16) * 2;
          for (int64_t kv2 = 0; kv2 < K2v; kv2++) {
            const int64_t n0 = 2 * kv2, n1 = 2 * kv2 + 1;
            base[kv2 * 128 + 0] = (n0 < Nkv) ? Vh[n0 * D + d] : 0;
            base[kv2 * 128 + 1] = (n1 < Nkv) ? Vh[n1 * D + d] : 0;
          }
        }
        // O[16][Npv] = P @ V
        for (int64_t dt = 0; dt < Npv / 64; dt++) {
          sme_tile_16x64_bf16(Av, Bv + dt * K2v * 128, O + dt * 64, K2v, Npv);
        }

        // ---- Write Out (bf16). ----
        for (int64_t i = 0; i < rows; i++) {
          uint16_t *orow = Oh + (m0 + i) * D;
          const float *crow = O + i * Npv;
          for (int64_t d = 0; d < D; d++) orow[d] = tle_f2bf(crow[d]);
        }
      }
    }

    free(Bk); free(Ak); free(S); free(P); free(Av); free(Bv); free(O);
  }
}

#else
EXPORT void sme_attn_prefill_bf16(const uint16_t *, const uint16_t *,
                                  const uint16_t *, uint16_t *, int64_t, int64_t,
                                  int64_t, float, int64_t, int64_t, int64_t) {}
#endif

}  // extern "C"
