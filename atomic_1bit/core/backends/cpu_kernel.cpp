#include "../kernel.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// --- SIMD INTRINSICS ---
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#endif

void ternary_matmul(const int8_t *A, const int8_t *B_transposed, int32_t *C,
                    int M, int N, int K) {
  // OPTIMIZATION NOTE:
  // This kernel ASSUMES 'B' is transposed to shape (N, K) [Row-Major].

#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {

      const int8_t *val_a = &A[i * K];
      const int8_t *val_b = &B_transposed[j * K]; // B is (N, K)

      int32_t total = 0;
      int k = 0;

#ifdef USE_NEON
      int32x4_t v_acc = vdupq_n_s32(0);

      // Process 16 items at a time
      for (; k <= K - 16; k += 16) {
        int8x16_t va = vld1q_s8(val_a + k);
        int8x16_t vb = vld1q_s8(val_b + k);

        // Ternary Multiply: (-1,0,1) * x => -x, 0, x
        // Standard integer multiply works perfectly.
        int8x16_t vprod = vmulq_s8(va, vb);

        // Accumulate: int8 -> int16 -> int32
        // Pairwise add and widen to int16
        int16x8_t v_pair = vpaddlq_s8(vprod);

        // Pairwise add and widen to int32
        int32x4_t v_quad = vpaddlq_s16(v_pair);

        // Accumulate to total vector
        v_acc = vaddq_s32(v_acc, v_quad);
      }

      // Horizontal sum
      total += vgetq_lane_s32(v_acc, 0);
      total += vgetq_lane_s32(v_acc, 1);
      total += vgetq_lane_s32(v_acc, 2);
      total += vgetq_lane_s32(v_acc, 3);

#elif defined(USE_AVX2)
      // AVX2 Implementation (32 bytes)
      __m256i v_acc = _mm256_setzero_si256();
      __m256i ones = _mm256_set1_epi16(1);

      for (; k <= K - 32; k += 32) {
        __m256i va = _mm256_loadu_si256((__m256i *)(val_a + k));
        __m256i vb = _mm256_loadu_si256((__m256i *)(val_b + k));

        // _mm256_sign_epi8(a, b):
        // Negates a if b < 0, Zeroes a if b == 0, Preserves a if b > 0.
        __m256i vprod = _mm256_sign_epi8(va, vb);

        __m256i v_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vprod));
        __m256i v_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vprod, 1));

        v_lo = _mm256_madd_epi16(v_lo, ones); // sums pairs to int32
        v_hi = _mm256_madd_epi16(v_hi, ones);

        v_acc = _mm256_add_epi32(v_acc, v_lo);
        v_acc = _mm256_add_epi32(v_acc, v_hi);
      }

      // Horizontal Sum of v_acc
      // Reduce 256 -> 128
      __m128i vlow = _mm256_castsi256_si128(v_acc);
      __m128i vhigh = _mm256_extracti128_si256(v_acc, 1);
      vlow = _mm_add_epi32(vlow, vhigh);
      // Scan sum
      int32_t tmp[4];
      _mm_storeu_si128((__m128i *)tmp, vlow);
      total += tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif

      // Tail loop (also handles non-SIMD)
      for (; k < K; ++k) {
        int8_t bv = val_b[k];
        if (bv == 1)
          total += val_a[k];
        else if (bv == -1)
          total -= val_a[k];
      }

      C[i * N + j] = total;
    }
  }
}
