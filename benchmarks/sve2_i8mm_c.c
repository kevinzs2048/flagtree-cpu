#define _POSIX_C_SOURCE 200809L

#include <arm_neon.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline uint64_t ns_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void pack_b_8x8(const int8_t *b, int ldb, int nb, int kb,
                              int8_t *dst) {
  int8x8_t r0 = vld1_s8(b + (size_t)(kb + 0) * ldb + nb);
  int8x8_t r1 = vld1_s8(b + (size_t)(kb + 1) * ldb + nb);
  int8x8_t r2 = vld1_s8(b + (size_t)(kb + 2) * ldb + nb);
  int8x8_t r3 = vld1_s8(b + (size_t)(kb + 3) * ldb + nb);
  int8x8_t r4 = vld1_s8(b + (size_t)(kb + 4) * ldb + nb);
  int8x8_t r5 = vld1_s8(b + (size_t)(kb + 5) * ldb + nb);
  int8x8_t r6 = vld1_s8(b + (size_t)(kb + 6) * ldb + nb);
  int8x8_t r7 = vld1_s8(b + (size_t)(kb + 7) * ldb + nb);

  int8x8x2_t t0 = vtrn_s8(r0, r1);
  int8x8x2_t t1 = vtrn_s8(r2, r3);
  int8x8x2_t t2 = vtrn_s8(r4, r5);
  int8x8x2_t t3 = vtrn_s8(r6, r7);
  int16x4x2_t u0 =
      vtrn_s16(vreinterpret_s16_s8(t0.val[0]),
               vreinterpret_s16_s8(t1.val[0]));
  int16x4x2_t u1 =
      vtrn_s16(vreinterpret_s16_s8(t0.val[1]),
               vreinterpret_s16_s8(t1.val[1]));
  int16x4x2_t u2 =
      vtrn_s16(vreinterpret_s16_s8(t2.val[0]),
               vreinterpret_s16_s8(t3.val[0]));
  int16x4x2_t u3 =
      vtrn_s16(vreinterpret_s16_s8(t2.val[1]),
               vreinterpret_s16_s8(t3.val[1]));
  int32x2x2_t v0 =
      vtrn_s32(vreinterpret_s32_s16(u0.val[0]),
               vreinterpret_s32_s16(u2.val[0]));
  int32x2x2_t v1 =
      vtrn_s32(vreinterpret_s32_s16(u1.val[0]),
               vreinterpret_s32_s16(u3.val[0]));
  int32x2x2_t v2 =
      vtrn_s32(vreinterpret_s32_s16(u0.val[1]),
               vreinterpret_s32_s16(u2.val[1]));
  int32x2x2_t v3 =
      vtrn_s32(vreinterpret_s32_s16(u1.val[1]),
               vreinterpret_s32_s16(u3.val[1]));

  vst1q_s8(dst + 0,
           vcombine_s8(vreinterpret_s8_s32(v0.val[0]),
                       vreinterpret_s8_s32(v1.val[0])));
  vst1q_s8(dst + 16,
           vcombine_s8(vreinterpret_s8_s32(v2.val[0]),
                       vreinterpret_s8_s32(v3.val[0])));
  vst1q_s8(dst + 32,
           vcombine_s8(vreinterpret_s8_s32(v0.val[1]),
                       vreinterpret_s8_s32(v1.val[1])));
  vst1q_s8(dst + 48,
           vcombine_s8(vreinterpret_s8_s32(v2.val[1]),
                       vreinterpret_s8_s32(v3.val[1])));
}

// Row-major s8 x row-major s8 -> row-major s32.  This is the same 8x8x8
// register blocking used by the Triton lowering.  B packing is deliberately
// inside the function (and therefore inside the timing) for an apples-to-
// apples comparison with a Triton kernel that receives ordinary row-major B.
__attribute__((noinline, visibility("default")))
void gemm_sve2_i8mm(const int8_t *restrict a, const int8_t *restrict b,
                    int32_t *restrict c, int m, int n, int k) {
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  int8_t *bpack = __builtin_alloca((size_t)(k / 8) * 4 * 16);

  for (int nb = 0; nb < n; nb += 8) {
    for (int kb = 0; kb < k; kb += 8) {
      int8_t *dst = bpack + (size_t)(kb / 8) * 64;
      pack_b_8x8(b, n, nb, kb, dst);
    }

    for (int mb = 0; mb < m; mb += 8) {
      svint32_t c00 = svdup_s32(0), c01 = svdup_s32(0);
      svint32_t c02 = svdup_s32(0), c03 = svdup_s32(0);
      svint32_t c10 = svdup_s32(0), c11 = svdup_s32(0);
      svint32_t c12 = svdup_s32(0), c13 = svdup_s32(0);
      svint32_t c20 = svdup_s32(0), c21 = svdup_s32(0);
      svint32_t c22 = svdup_s32(0), c23 = svdup_s32(0);
      svint32_t c30 = svdup_s32(0), c31 = svdup_s32(0);
      svint32_t c32 = svdup_s32(0), c33 = svdup_s32(0);

      for (int kb = 0; kb < k; kb += 8) {
        int8_t apack[4][16] __attribute__((aligned(16)));
        for (int pair = 0; pair < 4; ++pair) {
          memcpy(apack[pair], a + (size_t)(mb + pair * 2) * k + kb, 8);
          memcpy(apack[pair] + 8,
                 a + (size_t)(mb + pair * 2 + 1) * k + kb, 8);
        }
        const int8_t *bp = bpack + (size_t)(kb / 8) * 64;
        svint8_t a0 = svld1_s8(pg8, apack[0]);
        svint8_t a1 = svld1_s8(pg8, apack[1]);
        svint8_t a2 = svld1_s8(pg8, apack[2]);
        svint8_t a3 = svld1_s8(pg8, apack[3]);
        svint8_t b0 = svld1_s8(pg8, bp + 0);
        svint8_t b1 = svld1_s8(pg8, bp + 16);
        svint8_t b2 = svld1_s8(pg8, bp + 32);
        svint8_t b3 = svld1_s8(pg8, bp + 48);

        c00 = svmmla_s32(c00, a0, b0);
        c01 = svmmla_s32(c01, a0, b1);
        c02 = svmmla_s32(c02, a0, b2);
        c03 = svmmla_s32(c03, a0, b3);
        c10 = svmmla_s32(c10, a1, b0);
        c11 = svmmla_s32(c11, a1, b1);
        c12 = svmmla_s32(c12, a1, b2);
        c13 = svmmla_s32(c13, a1, b3);
        c20 = svmmla_s32(c20, a2, b0);
        c21 = svmmla_s32(c21, a2, b1);
        c22 = svmmla_s32(c22, a2, b2);
        c23 = svmmla_s32(c23, a2, b3);
        c30 = svmmla_s32(c30, a3, b0);
        c31 = svmmla_s32(c31, a3, b1);
        c32 = svmmla_s32(c32, a3, b2);
        c33 = svmmla_s32(c33, a3, b3);
      }

#define STORE_PAIR(RP, CP, V)                                                 \
  do {                                                                        \
    int32_t out[4] __attribute__((aligned(16)));                              \
    svst1_s32(pg32, out, (V));                                                \
    c[(size_t)(mb + (RP)*2) * n + nb + (CP)*2] = out[0];                     \
    c[(size_t)(mb + (RP)*2) * n + nb + (CP)*2 + 1] = out[1];                 \
    c[(size_t)(mb + (RP)*2 + 1) * n + nb + (CP)*2] = out[2];                 \
    c[(size_t)(mb + (RP)*2 + 1) * n + nb + (CP)*2 + 1] = out[3];             \
  } while (0)
      STORE_PAIR(0, 0, c00);
      STORE_PAIR(0, 1, c01);
      STORE_PAIR(0, 2, c02);
      STORE_PAIR(0, 3, c03);
      STORE_PAIR(1, 0, c10);
      STORE_PAIR(1, 1, c11);
      STORE_PAIR(1, 2, c12);
      STORE_PAIR(1, 3, c13);
      STORE_PAIR(2, 0, c20);
      STORE_PAIR(2, 1, c21);
      STORE_PAIR(2, 2, c22);
      STORE_PAIR(2, 3, c23);
      STORE_PAIR(3, 0, c30);
      STORE_PAIR(3, 1, c31);
      STORE_PAIR(3, 2, c32);
      STORE_PAIR(3, 3, c33);
#undef STORE_PAIR
    }
  }
}

static int check(const int8_t *a, const int8_t *b, const int32_t *c, int m,
                 int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int32_t want = 0;
      for (int p = 0; p < k; ++p)
        want += (int32_t)a[(size_t)i * k + p] *
                (int32_t)b[(size_t)p * n + j];
      if (c[(size_t)i * n + j] != want) {
        fprintf(stderr, "mismatch at (%d,%d): got %d want %d\n", i, j,
                c[(size_t)i * n + j], want);
        return 0;
      }
    }
  }
  return 1;
}

int main(int argc, char **argv) {
  int m = argc > 1 ? atoi(argv[1]) : 128;
  int n = argc > 2 ? atoi(argv[2]) : m;
  int k = argc > 3 ? atoi(argv[3]) : m;
  int iters = argc > 4 ? atoi(argv[4]) : 100;
  if (m <= 0 || n <= 0 || k <= 0 || (m | n | k) % 8) {
    fprintf(stderr, "M, N and K must be positive multiples of 8\n");
    return 2;
  }

  int8_t *a = aligned_alloc(64, (size_t)m * k);
  int8_t *b = aligned_alloc(64, (size_t)k * n);
  int32_t *c = aligned_alloc(64, (size_t)m * n * sizeof(*c));
  if (!a || !b || !c)
    return 2;
  for (size_t i = 0; i < (size_t)m * k; ++i)
    a[i] = (int8_t)((i * 17 + 13) % 255 - 127);
  for (size_t i = 0; i < (size_t)k * n; ++i)
    b[i] = (int8_t)((i * 29 + 7) % 255 - 127);

  gemm_sve2_i8mm(a, b, c, m, n, k);
  if (!check(a, b, c, m, n, k))
    return 1;
  for (int i = 0; i < 20; ++i)
    gemm_sve2_i8mm(a, b, c, m, n, k);
  uint64_t start = ns_now();
  for (int i = 0; i < iters; ++i)
    gemm_sve2_i8mm(a, b, c, m, n, k);
  double ms = (double)(ns_now() - start) / 1.0e6 / iters;
  double gops = 2.0 * m * n * k / ms / 1.0e6;
  printf("PASS C-SVE2 M=%d N=%d K=%d %.6f ms %.2f GOPS\n", m, n, k, ms,
         gops);
  free(a);
  free(b);
  free(c);
  return 0;
}
