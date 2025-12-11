#ifndef ATOMIC_KERNEL_H
#define ATOMIC_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Performs matrix multiplication A * B = C
// A: (M x K) - input activations (int8)
// B: (K x N) - weights (int8, values -1, 0, 1)
// C: (M x N) - output (int32)
//
// Note: This is a "simulated" ternary kernel using int8 input for weights.
// Real BitNet would pack weights (2 bits), but logic remains:
// only add/sub allowed for calculation.
void ternary_matmul(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif // ATOMIC_KERNEL_H
