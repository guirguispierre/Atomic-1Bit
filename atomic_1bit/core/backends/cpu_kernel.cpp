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
      int32x4_t v_acc2 = vdupq_n_s32(0);
      int32x4_t v_acc3 = vdupq_n_s32(0);

#if defined(__ARM_FEATURE_DOTPROD)
      // vdotq_s32 (sdot, ARMv8.4+ / optional 8.2-8.3): each lane computes
      // acc[i] += a[4i]*b[4i] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] +
      // a[4i+3]*b[4i+3], widening every int8 product into int32 first, so
      // activation -128 against weight -1 gives +128 exactly (no int8
      // wraparound). Integer adds are associative, so splitting the dot
      // product across four accumulator lanes stays bit-exact.

      // Process 64 items at a time (4 independent accumulators)
      for (; k <= K - 64; k += 64) {
        v_acc0 = vdotq_s32(v_acc0, vld1q_s8(val_a + k),
                           vld1q_s8(val_b + k));
        v_acc1 = vdotq_s32(v_acc1, vld1q_s8(val_a + k + 16),
                           vld1q_s8(val_b + k + 16));
        v_acc2 = vdotq_s32(v_acc2, vld1q_s8(val_a + k + 32),
                           vld1q_s8(val_b + k + 32));
        v_acc3 = vdotq_s32(v_acc3, vld1q_s8(val_a + k + 48),
                           vld1q_s8(val_b + k + 48));
      }

      // Handle remaining 16-element blocks
      for (; k <= K - 16; k += 16) {
        v_acc0 = vdotq_s32(v_acc0, vld1q_s8(val_a + k),
                           vld1q_s8(val_b + k));
      }
#else
      // guirguispierre 2026-08-10 - vmull_s8 widens to int16 because vmulq_s8
      // truncates: activation -128 against weight -1 wraps and flips the sum

      // Process 32 items at a time (2x unrolled from 16)
      for (; k <= K - 32; k += 32) {
        // First 16 elements
        int8x16_t va0 = vld1q_s8(val_a + k);
        int8x16_t vb0 = vld1q_s8(val_b + k);
        v_acc0 = vaddq_s32(
            v_acc0, vpaddlq_s16(vmull_s8(vget_low_s8(va0), vget_low_s8(vb0))));
        v_acc0 = vaddq_s32(
            v_acc0, vpaddlq_s16(vmull_s8(vget_high_s8(va0), vget_high_s8(vb0))));

        // Second 16 elements
        int8x16_t va1 = vld1q_s8(val_a + k + 16);
        int8x16_t vb1 = vld1q_s8(val_b + k + 16);
        v_acc1 = vaddq_s32(
            v_acc1, vpaddlq_s16(vmull_s8(vget_low_s8(va1), vget_low_s8(vb1))));
        v_acc1 = vaddq_s32(
            v_acc1, vpaddlq_s16(vmull_s8(vget_high_s8(va1), vget_high_s8(vb1))));
      }

      // Handle remaining 16-element block
      for (; k <= K - 16; k += 16) {
        int8x16_t va = vld1q_s8(val_a + k);
        int8x16_t vb = vld1q_s8(val_b + k);
        v_acc0 = vaddq_s32(
            v_acc0, vpaddlq_s16(vmull_s8(vget_low_s8(va), vget_low_s8(vb))));
        v_acc0 = vaddq_s32(
            v_acc0, vpaddlq_s16(vmull_s8(vget_high_s8(va), vget_high_s8(vb))));
      }
#endif

      // Combine accumulators (unused lanes are zero in the fallback path)
      v_acc0 = vaddq_s32(v_acc0, v_acc1);
      v_acc2 = vaddq_s32(v_acc2, v_acc3);
      v_acc0 = vaddq_s32(v_acc0, v_acc2);

      // Horizontal sum
      total += vgetq_lane_s32(v_acc0, 0);
      total += vgetq_lane_s32(v_acc0, 1);
      total += vgetq_lane_s32(v_acc0, 2);
      total += vgetq_lane_s32(v_acc0, 3);

#elif defined(USE_AVX2)
      // AVX2 Implementation: 2x unrolled (64 bytes per iteration)
      __m256i v_acc0 = _mm256_setzero_si256();
      __m256i v_acc1 = _mm256_setzero_si256();
      const __m256i ones = _mm256_set1_epi16(1);

      // Widen to int16 and multiply instead of _mm256_sign_epi8: sign_epi8
      // computes a * sign(b) in int8, so activation -128 against weight -1
      // wraps back to -128 and flips the sum. int16 products cannot
      // overflow (|a*b| <= 16384) and madd_epi16 widens pairs into int32.
      // Integer adds are associative, so lane-wise reordering keeps the
      // result bit-exact vs the scalar loop.
      auto dot32 = [&](const int8_t *pa, const int8_t *pb, __m256i &acc) {
        const __m256i va = _mm256_loadu_si256((const __m256i *)pa);
        const __m256i vb = _mm256_loadu_si256((const __m256i *)pb);
        const __m256i lo = _mm256_mullo_epi16(
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va)),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb)));
        const __m256i hi = _mm256_mullo_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1)),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo, ones));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(hi, ones));
      };

      for (; k <= K - 64; k += 64) {
        dot32(val_a + k, val_b + k, v_acc0);
        dot32(val_a + k + 32, val_b + k + 32, v_acc1);
      }

      // Remaining 32-element block
      for (; k <= K - 32; k += 32) {
        dot32(val_a + k, val_b + k, v_acc0);
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

      // Scalar tail (also the whole loop on non-SIMD builds). Branch-free
      // multiply: int8 operands promote to int, so (-128) x (-1) = +128 is
      // exact, and weights outside {-1,0,1} still produce the true product.
      for (; k < K; ++k) {
        total += (int32_t)val_a[k] * (int32_t)val_b[k];
      }

      C[i * N + j] = total;
    }
  }
}
