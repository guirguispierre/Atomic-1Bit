#include "kernel.h"
#include <stdio.h>

// "The Magic Kernel"
// Naive CPU implementation of Ternary Matmul (Linear Layer)
// Optimized for readability and proof of concept.
//
// No multiplication. Only Addition and Subtraction.
void ternary_matmul(const int8_t *A, const int8_t *B, int32_t *C, int M, int N,
                    int K) {
  // A is M x K
  // B is N x K (Transposed for memory locality usually? Or standard K x N?)
  // Let's assume standard K x N for now to match standard math notation,
  // but typical inference engines prefer B to be (N, K) for row-major access.
  // Let's assume K x N for simplicity of "A * B".
  //
  // Actually, for cache efficiency, iterating weights linearly is better if
  // Output channel is inner loop? Let's stick to standard naive loops for v1
  // "Bare Metal" readablity.
  //
  // i: 0..M (Batch size)
  // j: 0..N (Output Dim)
  // k: 0..K (Input Dim / Inner Dim)

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        // Feature from A
        int8_t a_val = A[i * K + k]; // A[i][k]

        // Weight from B
        // B is K x N -> B[k * N + j]
        int8_t w_val = B[k * N + j];

        // The "Atomic" Operation
        if (w_val == 1) {
          acc += a_val;
        } else if (w_val == -1) {
          acc -= a_val;
        }
        // Implicit else: w_val == 0 -> do nothing (sparsity!)
      }
      C[i * N + j] = acc;
    }
  }
}
