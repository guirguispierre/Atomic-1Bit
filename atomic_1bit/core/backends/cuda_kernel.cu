#include "../kernel.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Tile size for shared-memory tiled matrix multiplication.
// Each thread block is TILE_SIZE x TILE_SIZE threads and processes a
// TILE_SIZE x TILE_SIZE sub-tile of the output matrix C per iteration.
// 16 gives 256 threads per block, which maps well to warp granularity (32
// threads/warp -> 8 warps/block) and fits shared memory constraints.
#define TILE_SIZE 16

// ---------------------------------------------------------------------------
// ternary_matmul_kernel
//
// Purpose:
//   Computes C = A * B^T using tiled shared-memory matrix multiplication,
//   where A and B contain ternary (int8) values.  The operation is:
//     C[i, j] = sum_k  A[i, k] * B[j, k]
//   which is equivalent to C = A * B^T when B is stored row-major as (N, K).
//
// Parameters:
//   A   - Input activation matrix, shape (M, K), row-major int8.
//         Typically a batch of quantized INT8 activations.
//   B   - Weight matrix stored as (N, K), row-major int8, values in {-1, 0, 1}.
//         This is the ternary weight matrix already transposed for efficient
//         dot-product access (each row of B is one output neuron's weights).
//   C   - Output accumulation matrix, shape (M, N), row-major int32.
//         The caller must allocate this buffer before the kernel launches.
//   M   - Number of rows in A (batch dimension / sequence positions).
//   N   - Number of rows in B (output feature dimension).
//   K   - Shared inner dimension (input feature dimension).
//
// Launch configuration:
//   Grid:  (ceil(N / TILE_SIZE), ceil(M / TILE_SIZE))
//            -> grid.x covers the N (output) dimension
//            -> grid.y covers the M (batch/row) dimension
//   Block: (TILE_SIZE, TILE_SIZE)  =  16 x 16 = 256 threads per block
//            -> threadIdx.x covers columns (K-tile index and output column)
//            -> threadIdx.y covers rows    (K-tile index and output row)
//
// Shared memory:
//   As[TILE_SIZE][TILE_SIZE] - Tile of A loaded cooperatively by the block.
//                               As[ty][tx] = A[row, k_tile_start + tx]
//   Bs[TILE_SIZE][TILE_SIZE] - Tile of B loaded cooperatively by the block.
//                               Bs[tx][ty] = B[col, k_tile_start + ty]
//                               (Transposed load into shared memory so the
//                               inner-loop access Bs[tx][p] is coalesced for
//                               all threads sharing the same tx value.)
//
// Thread/block mapping:
//   Each thread (tx, ty) in block (bx, by) is responsible for one output
//   element: C[by*TILE + ty, bx*TILE + tx].  The thread accumulates its
//   partial dot product across all K-tiles into the register variable 'val'
//   and writes the result to global memory after all tiles are processed.
//
// Synchronization:
//   __syncthreads() is called twice per K-tile:
//     1. After loading As and Bs to ensure all threads see the complete tile
//        before the inner product loop begins.
//     2. After the inner product loop to prevent any thread from overwriting
//        the shared tile before every thread has finished reading it.
// ---------------------------------------------------------------------------
__global__ void ternary_matmul_kernel(const int8_t *A, const int8_t *B,
                                      int32_t *C, int M, int N, int K) {
  // Global column and row indices for the output element this thread computes.
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Row in A (and C) that this thread is responsible for.
  int row = by * TILE_SIZE + ty;
  // Column in C (and row in B, since B is transposed) for this thread.
  int col = bx * TILE_SIZE + tx;

  // Running int32 accumulator for the dot product C[row, col].
  // Using int32 prevents overflow when accumulating up to K int8*int8 products.
  int32_t val = 0;

  // Shared memory tiles.
  // As[ty][tx]: one TILE_SIZE x TILE_SIZE slice of A for the current K-tile.
  // Bs[tx][ty]: one TILE_SIZE x TILE_SIZE slice of B for the current K-tile,
  //             stored transposed to make the accumulation loop access
  //             Bs[tx][p] with tx fixed per thread (no bank conflicts on p).
  __shared__ int8_t As[TILE_SIZE][TILE_SIZE];
  __shared__ int8_t Bs[TILE_SIZE][TILE_SIZE];

  // Iterate over K in steps of TILE_SIZE, loading and processing one tile at
  // a time.  ceil(K / TILE_SIZE) iterations handle non-multiples of TILE_SIZE
  // via boundary guards below.
  for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {

    // --- Load tile of A into shared memory ---
    // Thread (ty, tx) loads A[row, k*TILE + tx].
    // k_idx_a is the global K-dimension index for this thread's A element.
    int k_idx_a = k * TILE_SIZE + tx;
    if (row < M && k_idx_a < K)
      As[ty][tx] = A[row * K + k_idx_a];
    else
      As[ty][tx] = 0; // Zero-pad out-of-bounds elements.

    // --- Load tile of B into shared memory (transposed) ---
    // B is stored as (N, K): B[col, k_idx_b] is the weight for output neuron
    // 'col' at input feature 'k_idx_b'.
    // We load B[col, k*TILE + ty] into Bs[tx][ty].
    // This transposed indexing (Bs[tx][ty] rather than Bs[ty][tx]) ensures
    // that the accumulation loop below can read Bs[tx][p] with tx fixed for
    // each thread, keeping shared memory access patterns efficient.
    int k_idx_b = k * TILE_SIZE + ty; // ty indexes into the K dimension of B
    if (col < N && k_idx_b < K)
      Bs[tx][ty] = B[col * K + k_idx_b];
    else
      Bs[tx][ty] = 0; // Zero-pad out-of-bounds elements.

    // Barrier 1: All threads in the block must finish loading their tile
    // elements into shared memory before any thread begins the dot product.
    __syncthreads();

    // --- Inner product over the loaded tile ---
    // val += A[row, k_base + p] * B[col, k_base + p]  for p in [0, TILE_SIZE)
    // As[ty][p] provides A[row, k_base + p].
    // Bs[tx][p] provides B[col, k_base + p] (due to the transposed load above).
    for (int p = 0; p < TILE_SIZE; ++p) {
      // Optimization: Ternary logic
      // int8 multiplication is fast, but we can also use branching if needed.
      // On GPU, multiplication is usually better than divergence.
      // But here val is int32, As/Bs are int8.
      val +=
          (int32_t)As[ty][p] *
          (int32_t)Bs[tx][p]; // Accessing Bs[tx][p] corresponds to B[col, k+p]
    }

    // Barrier 2: All threads must finish consuming the shared tile before the
    // next iteration overwrites As and Bs with the next K-tile.
    __syncthreads();
  }

  // Write the accumulated result to global memory.
  // Guard ensures out-of-bounds threads (from grid padding) do not write.
  if (row < M && col < N) {
    C[row * N + col] = val;
  }
}

// ---------------------------------------------------------------------------
// ternary_matmul  (C-linkage wrapper)
//
// Purpose:
//   Host-side entry point that manages device memory allocation, data
//   transfers, kernel launch, result copy-back, and cleanup for a single
//   ternary matrix multiplication call.  This is the symbol loaded at runtime
//   by the Python ctypes wrapper (atomic_1bit/python/wrapper.py).
//
// Parameters:
//   A             - Host pointer to INT8 activation matrix, shape (M, K).
//   B_transposed  - Host pointer to INT8 weight matrix, shape (N, K).
//                   The caller (Python wrapper) is responsible for transposing
//                   the standard (K, N) weight layout to (N, K) before calling.
//   C             - Host pointer to INT32 output buffer, shape (M, N).
//                   Must be allocated by the caller before this function.
//   M, N, K       - Matrix dimensions as described above.
//
// Notes:
//   - Device memory is allocated, used, and freed on every call.  This is
//     acceptable for a proof-of-concept but adds significant overhead for
//     repeated calls.  A production backend would use persistent device
//     buffers or the stream-based API.
//   - The CUDA runtime implicitly manages the CUDA context; no explicit
//     cuInit / cuCtxCreate is required.
//   - Error checking on CUDA API calls is omitted for brevity in this
//     proof-of-concept implementation (v1.2).
// ---------------------------------------------------------------------------
extern "C" {
void ternary_matmul(const int8_t *A, const int8_t *B_transposed, int32_t *C,
                    int M, int N, int K) {
  // Device pointers for A, B, and C.
  int8_t *d_A, *d_B;
  int32_t *d_C;

  // Allocate device memory for all three matrices.
  cudaMalloc(&d_A, M * K * sizeof(int8_t));
  cudaMalloc(&d_B, N * K * sizeof(int8_t));
  cudaMalloc(&d_C, M * N * sizeof(int32_t));

  // Copy input matrices from host to device.
  cudaMemcpy(d_A, A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B_transposed, N * K * sizeof(int8_t), cudaMemcpyHostToDevice);

  // Configure launch parameters.
  // Block: 16x16 = 256 threads.
  // Grid: enough blocks to cover the full (M, N) output matrix, with one
  //       thread per output element.  Padding blocks handle non-multiples of
  //       TILE_SIZE via the boundary guards inside the kernel.
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  // Launch the kernel.  Execution is asynchronous with respect to the host;
  // the following cudaMemcpy implicitly synchronizes.
  ternary_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  // Copy result back from device to host (synchronizes the kernel).
  cudaMemcpy(C, d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

  // Free device memory.
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Note: Re-allocating every call is slow.
  // Ideally we would use persistent buffers or the stream-based API in a real
  // backend. For v1.2 proof-of-concept, this is sufficient.
}
}
