#ifndef ATOMIC_KERNEL_H
#define ATOMIC_KERNEL_H

#include <stdint.h>

extern "C" {

// Optimized Ternary Matmul
// C = A * B^T
// A: (M, K)
// B_transposed: (N, K) -> Stored row-major (N, K) effectively representing B^T
// if B was (K, N) Actually, in PyTorch W is (Out, In). So B here is Weights. We
// expect Weights to be stored as (N, K) contiguous. Output C is (M, N).
void ternary_matmul(const int8_t *A, const int8_t *B, int32_t *C, int M, int N,
                    int K);
}

#endif
