#include <metal_stdlib>
using namespace metal;

// Ternary Matrix Multiplication Kernel
// A: M x K (int8)
// B: N x K (int8) - Transposed layout
// C: M x N (int32)
//
// Thread (row, col) calculates C[row, col]
kernel void ternary_matmul_shader(
    const device char *A [[ buffer(0) ]],
    const device char *B [[ buffer(1) ]],
    device int *C [[ buffer(2) ]],
    constant int &M [[ buffer(3) ]],
    constant int &N [[ buffer(4) ]],
    constant int &K [[ buffer(5) ]],
    uint2 id [[ thread_position_in_grid ]]
) {
    if (id.x >= (uint)N || id.y >= (uint)M) {
        return;
    }

    // A is (M, K) -> Row Major -> A[row * K + k]
    // B is (N, K) -> Row Major -> B[col * K + k]
    
    int row = id.y;
    int col = id.x;
    
    int32_t acc = 0;
    
    // Pointers to the start of the row/col
    const device char* row_A = A + row * K;
    const device char* row_B = B + col * K; // B is transposed (N,K)

    // Vectorized loop (4 bytes at a time)
    // char4 is 4 bytes. 
    int k = 0;
    for (; k <= K - 4; k += 4) {
        char4 val_a = *((const device char4*)(row_A + k));
        char4 val_b = *((const device char4*)(row_B + k));
        
        // Manual unroll or let compiler handle it
        // val_b components: -1, 0, 1
        
        // Element 0
        if (val_b.x == 1) acc += val_a.x;
        else if (val_b.x == -1) acc -= val_a.x;
        
        // Element 1
        if (val_b.y == 1) acc += val_a.y;
        else if (val_b.y == -1) acc -= val_a.y;
        
        // Element 2
        if (val_b.z == 1) acc += val_a.z;
        else if (val_b.z == -1) acc -= val_a.z;
        
        // Element 3
        if (val_b.w == 1) acc += val_a.w;
        else if (val_b.w == -1) acc -= val_a.w;
    }
    
    // Tail
    for (; k < K; ++k) {
        char val_a = row_A[k];
        char val_b = row_B[k];
        
        if (val_b == 1) acc += val_a;
        else if (val_b == -1) acc -= val_a;
    }

    C[row * N + col] = acc;
}
