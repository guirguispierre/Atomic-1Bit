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
      int32x4_t v_acc0 = vdupq_n_s32(0);
      int32x4_t v_acc1 = vdupq_n_s32(0);

      // Process 32 items at a time (2x unrolled from 16)
      for (; k <= K - 32; k += 32) {
        // First 16 elements
        int8x16_t va0 = vld1q_s8(val_a + k);
        int8x16_t vb0 = vld1q_s8(val_b + k);
        int8x16_t vprod0 = vmulq_s8(va0, vb0);
        int16x8_t v_pair0 = vpaddlq_s8(vprod0);
        int32x4_t v_quad0 = vpaddlq_s16(v_pair0);
        v_acc0 = vaddq_s32(v_acc0, v_quad0);

        // Second 16 elements
        int8x16_t va1 = vld1q_s8(val_a + k + 16);
        int8x16_t vb1 = vld1q_s8(val_b + k + 16);
        int8x16_t vprod1 = vmulq_s8(va1, vb1);
        int16x8_t v_pair1 = vpaddlq_s8(vprod1);
        int32x4_t v_quad1 = vpaddlq_s16(v_pair1);
        v_acc1 = vaddq_s32(v_acc1, v_quad1);
      }

      // Handle remaining 16-element block
      for (; k <= K - 16; k += 16) {
        int8x16_t va = vld1q_s8(val_a + k);
        int8x16_t vb = vld1q_s8(val_b + k);
        int8x16_t vprod = vmulq_s8(va, vb);
        int16x8_t v_pair = vpaddlq_s8(vprod);
        int32x4_t v_quad = vpaddlq_s16(v_pair);
        v_acc0 = vaddq_s32(v_acc0, v_quad);
      }

      // Combine accumulators
      v_acc0 = vaddq_s32(v_acc0, v_acc1);

      // Horizontal sum
      total += vgetq_lane_s32(v_acc0, 0);
      total += vgetq_lane_s32(v_acc0, 1);
      total += vgetq_lane_s32(v_acc0, 2);
      total += vgetq_lane_s32(v_acc0, 3);

#elif defined(USE_AVX2)
      // AVX2 Implementation: 2x unrolled (64 bytes per iteration)
      __m256i v_acc0 = _mm256_setzero_si256();
      __m256i v_acc1 = _mm256_setzero_si256();
      __m256i ones = _mm256_set1_epi16(1);

      for (; k <= K - 64; k += 64) {
        // First 32 elements
        __m256i va0 = _mm256_loadu_si256((__m256i *)(val_a + k));
        __m256i vb0 = _mm256_loadu_si256((__m256i *)(val_b + k));
        __m256i vprod0 = _mm256_sign_epi8(va0, vb0);
        __m256i v_lo0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vprod0));
        __m256i v_hi0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vprod0, 1));
        v_lo0 = _mm256_madd_epi16(v_lo0, ones);
        v_hi0 = _mm256_madd_epi16(v_hi0, ones);
        v_acc0 = _mm256_add_epi32(v_acc0, v_lo0);
        v_acc0 = _mm256_add_epi32(v_acc0, v_hi0);

        // Second 32 elements
        __m256i va1 = _mm256_loadu_si256((__m256i *)(val_a + k + 32));
        __m256i vb1 = _mm256_loadu_si256((__m256i *)(val_b + k + 32));
        __m256i vprod1 = _mm256_sign_epi8(va1, vb1);
        __m256i v_lo1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vprod1));
        __m256i v_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vprod1, 1));
        v_lo1 = _mm256_madd_epi16(v_lo1, ones);
        v_hi1 = _mm256_madd_epi16(v_hi1, ones);
        v_acc1 = _mm256_add_epi32(v_acc1, v_lo1);
        v_acc1 = _mm256_add_epi32(v_acc1, v_hi1);
      }

      // Remaining 32-element block
      for (; k <= K - 32; k += 32) {
        __m256i va = _mm256_loadu_si256((__m256i *)(val_a + k));
        __m256i vb = _mm256_loadu_si256((__m256i *)(val_b + k));
        __m256i vprod = _mm256_sign_epi8(va, vb);
        __m256i v_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vprod));
        __m256i v_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vprod, 1));
        v_lo = _mm256_madd_epi16(v_lo, ones);
        v_hi = _mm256_madd_epi16(v_hi, ones);
        v_acc0 = _mm256_add_epi32(v_acc0, v_lo);
        v_acc0 = _mm256_add_epi32(v_acc0, v_hi);
      }

      // Combine accumulators
      v_acc0 = _mm256_add_epi32(v_acc0, v_acc1);

      // Horizontal Sum
      __m128i vlow = _mm256_castsi256_si128(v_acc0);
      __m128i vhigh = _mm256_extracti128_si256(v_acc0, 1);
      vlow = _mm_add_epi32(vlow, vhigh);
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
