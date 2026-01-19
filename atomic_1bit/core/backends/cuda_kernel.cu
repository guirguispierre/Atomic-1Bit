#include "../kernel.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define TILE_SIZE 16

// CUDA Kernel: Tiled Matrix Multiplication
// A: (M, K)
// B: (N, K) transposed -> stored as (N, K) naturally
// C: (M, N)
//
// We want C[i, j] = sum(A[i, k] * B[j, k]) over k.
// B is (N, K), so row j of C corresponds to row j of B.
// This is actually: C = A * B^T
//
// Grid: (ceil(N/TILE), ceil(M/TILE))
// Block: (TILE, TILE)
__global__ void ternary_matmul_kernel(const int8_t *A, const int8_t *B,
                                      int32_t *C, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Row and Col for C
  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  int32_t val = 0;

  // Shared memory for tiles
  // dim A: M x K
  // dim B: N x K
  // Tile A: (TILE, TILE) -> subsection of A
  // Tile B: (TILE, TILE) -> subsection of B
  // Loop over K in steps of TILE_SIZE
  __shared__ int8_t As[TILE_SIZE][TILE_SIZE];
  __shared__ int8_t Bs[TILE_SIZE][TILE_SIZE];

  for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {

    // Load A tile
    // A element: [row, k * TILE + tx]
    // Guard boundaries
    int k_idx = k * TILE_SIZE + tx;
    if (row < M && k_idx < K)
      As[ty][tx] = A[row * K + k_idx];
    else
      As[ty][tx] = 0;

    // Load B tile
    // B element: [col, k * TILE + ty]
    // Note: B is (N, K). We need row 'col' of B, and column 'p' of B?
    // Wait, C[row, col] = dot(A_row, B_col_of_transpose) = dot(A_row,
    // B_row_of_original). Since B is passed as (N, K), it IS B_transposed
    // relative to math A*B. So we need dot(A[row, :], B[col, :]).
    //
    // Loading B:
    // We want B[col, current_k].
    // inside tile:
    // B tile needs to provide values for the dot product phase.
    // Dot product is over 'k'.
    // Thread (ty, tx) computes C[row, col].
    // We accumulate over k.
    // So we need A[row, k] and B[col, k].
    //
    // In shared memory:
    // As[ty][t_k] -> A element for this thread's row, at inner index t_k
    // Bs[tx][t_k] -> B element for this thread's col, at inner index t_k
    //
    // Let's load:
    // As[ty][tx] corresponds to A[row, k_start + tx]. (Loading a chunk of row)
    // Bs[tx][ty] corresponds to B[col, k_start + ty]???
    // No, standard tiling loads sub-blocks.
    //
    // Let's stick to standard loading:
    // Load A[row, k*TILE + tx] -> As[ty][tx]
    // Load B[col, k*TILE + ty] -> Bs[tx][ty]  <-- Transposed load into shared
    // memory to optimize access? Or just load B[col, k*TILE + tx]?
    //
    // If we load B[col, k*TILE + tx] into Bs[ty][tx] (mapping threads to B tile
    // naturally): B element: B[col, k_idx] k_idx = k * TILE_SIZE + tx? No, we
    // need TILE x TILE block.
    //
    // Let's load B[col, k*TILE + tx] -> Bs[ty][tx]  (Thread mapping:
    // ty->row(col actually), tx->k) Wait, 'col' depends on 'bx' and 'tx'. 'row'
    // depends on 'by' and 'ty'. This thread (tx, ty) computes C[row, col].
    //
    // Inside computation loop (p = 0..TILE-1):
    //   val += As[ty][p] * Bs[?][p]
    // We need B[col, current_k_p].
    // If we loaded B into Bs s.t. Bs[tx][p] is B[col, p]... confusing.
    //
    // Standard Matrix Mul A x B:
    // C[y, x] = sum A[y, k] * B[k, x]
    // Here we have A x B^T.
    // C[row, col] = sum A[row, k] * B[col, k]
    //
    // Let's load A chunk: A[row, k_chunk + tx].
    // Let's load B chunk: B[col, k_chunk + ty].
    //
    // Issue: k_chunk + tx means we are loading different k's for the same row.
    // Yes, we cooperatively load the tile A[row_range, k_range].
    // TILE=16. Block=16x16.
    // We need to load A[row, k_chunk..k_chunk+15].
    // 256 threads.
    // Each thread loads 1 element of A tile and 1 element of B tile.
    // As[ty][tx] = A[row, k*TILE + tx].
    // Bs[ty][tx] = B[col, k*TILE + tx] (Since B is (N, K), 'col' is first dim).

    int k_curr = k * TILE_SIZE + tx;
    if (col < N && k_curr < K)
      Bs[ty][tx] = B[col * K + k_curr];
    else
      Bs[ty][tx] = 0;

    __syncthreads();

    // Compute
    // val += A[row, k_base + p] * B[col, k_base + p]
    // As[ty][p] is A[row, k_base + p]
    // Bs[tx][p] is B[col, k_base + p] ??? No.
    // Bs[ty][tx] stored B[col, k+tx].
    // So Bs[tx][p] (if we use tx as col index inside block) would give B[?, ?].
    //
    // My load was: Bs[ty][tx] = B[col, k+tx].
    // 'col' is (bx*TILE + ??). 'col' in my load logic was (bx*TILE + tx).
    // Wait, earlier I defined col = bx * TILE + tx.
    // So Bs[ty][tx] holds B[col_of_thread, k+tx].
    // This seems wrong for what we need. We need B[col_of_this_thread, k+p].
    // But B is shared.
    // Every thread in this block needs B[its_col, k+p].
    // 'its_col' is fixed for the thread (bx*TILE + tx).
    // So Bs needs to store [tx][p].
    //
    // Let's verify loading:
    // Bs[tx][ty] = B[ (bx*TILE + tx), (k*TILE + ty) ]
    // -> Loads B[col, k] column-major into shared mem?
    //
    // Let's stick to simplest coordinate mapping:
    // As[ty][tx] = A[row, k*TILE + tx]
    // Bs[tx][ty] = B[col, k*TILE + ty]  <-- Load transposed B tile?
    //
    // Let's try:
    // int t_row = row;       // global row index for A
    // int t_col = col;       // global col index for B (which is row index in B
    // tensor)
    //
    // As[ty][tx] = A[t_row, k*TILE + tx];
    // Bs[tx][ty] = B[t_col, k*TILE + ty];  // Load B[col, k]
    //
    // Boundary checks...
    //
    // Then loop p=0..TILE-1:
    // val += As[ty][p] * Bs[tx][p];
    // As[ty][p] -> A[row, k+p]
    // Bs[tx][p] -> B[col, k+p]
    // Correct!

    // Correct Loading Logic:
    int k_idx_a = k * TILE_SIZE + tx;
    if (row < M && k_idx_a < K)
      As[ty][tx] = A[row * K + k_idx_a];
    else
      As[ty][tx] = 0;

    int k_idx_b = k * TILE_SIZE + ty; // we use ty for k dimension in B load
    // We use 'col' (bx*TILE + tx) as the 'N' dimension index for B.
    if (col < N && k_idx_b < K)
      Bs[tx][ty] = B[col * K + k_idx_b];
    else
      Bs[tx][ty] = 0;

    __syncthreads();

    for (int p = 0; p < TILE_SIZE; ++p) {
      // Optimization: Ternary logic
      // int8 multiplication is fast, but we can also use branching if needed.
      // On GPU, multiplication is usually better than divergence.
      // But here val is int32, As/Bs are int8.
      val +=
          (int32_t)As[ty][p] *
          (int32_t)Bs[tx][p]; // Accessing Bs[tx][p] corresponds to B[col, k+p]
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = val;
  }
}

// Wrapper to launch kernel
extern "C" {
void ternary_matmul(const int8_t *A, const int8_t *B_transposed, int32_t *C,
                    int M, int N, int K) {
  // Implement lazy init or assume context is set?
  // Standard CUDA runtime API handles context.

  // Allocate device memory
  int8_t *d_A, *d_B;
  int32_t *d_C;

  cudaMalloc(&d_A, M * K * sizeof(int8_t));
  cudaMalloc(&d_B, N * K * sizeof(int8_t));
  cudaMalloc(&d_C, M * N * sizeof(int32_t));

  // Copy inputs
  cudaMemcpy(d_A, A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B_transposed, N * K * sizeof(int8_t), cudaMemcpyHostToDevice);

  // Launch
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  ternary_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  // Copy back
  cudaMemcpy(C, d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

  // Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Note: Re-allocating every call is slow.
  // Ideally we would use persistent buffers or the stream-based API in a real
  // backend. For v1.2 proof-of-concept, this is sufficient.
}
}
