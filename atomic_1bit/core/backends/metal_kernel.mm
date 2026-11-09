#include "../kernel.h"
#include <Metal/Metal.h>
#include <dlfcn.h>
#include <iostream>
#include <string.h>

// Scalar CPU fallback — identical semantics to the CPU backend's tail loop.
// Used when no Metal device exists, when default.metallib cannot be found,
// or when a zero-copy buffer wrap fails. Never exits the host process.
static void ternary_matmul_cpu(const int8_t *A, const int8_t *B_transposed,
                               int32_t *C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      const int8_t *val_a = &A[i * K];
      const int8_t *val_b = &B_transposed[j * K];
      int32_t total = 0;
      for (int k = 0; k < K; ++k) {
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

// Singleton for Metal Context
struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> commandQueue = nil;
  id<MTLComputePipelineState> pipelineState = nil;
  bool ok = false;

  MetalContext() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::cerr << "Metal backend: no Metal device found, using CPU fallback."
                << std::endl;
      return;
    }
    commandQueue = [device newCommandQueue];

    // Load default library
    NSError *error = nil;

    // default.metallib sits next to this dylib. Resolve the dylib's own path
    // at runtime with dladdr() — robust regardless of the compile-time cwd
    // (__FILE__ is only a relative path when built via the Makefile).
    id<MTLLibrary> defaultLibrary = nil;
    Dl_info info;
    if (dladdr((const void *)this, &info) && info.dli_fname) {
      NSString *dylibDir = [[NSString
          stringWithUTF8String:info.dli_fname] stringByDeletingLastPathComponent];
      NSURL *libOnDisk = [NSURL
          fileURLWithPath:[dylibDir
                              stringByAppendingPathComponent:@"default.metallib"]];
      defaultLibrary = [device newLibraryWithURL:libOnDisk error:&error];
    }
    if (!defaultLibrary) {
      // Fallback: try "default.metallib" relative to cwd (e.g. when running
      // from atomic_1bit/core/ directly during development).
      defaultLibrary = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@"default.metallib"]
                      error:&error];
    }

    if (!defaultLibrary) {
      std::cerr << "Metal backend: could not load default.metallib ("
                << [[error localizedDescription] UTF8String]
                << "), using CPU fallback." << std::endl;
      return;
    }

    id<MTLFunction> kernelFunction =
        [defaultLibrary newFunctionWithName:@"ternary_matmul_shader"];
    if (!kernelFunction) {
      std::cerr << "Metal backend: 'ternary_matmul_shader' not in library, "
                   "using CPU fallback."
                << std::endl;
      return;
    }

    pipelineState = [device newComputePipelineStateWithFunction:kernelFunction
                                                          error:&error];
    if (!pipelineState) {
      std::cerr << "Metal backend: could not create pipeline state ("
                << [[error localizedDescription] UTF8String]
                << "), using CPU fallback." << std::endl;
      return;
    }

    ok = true;
  }
};

static MetalContext *g_metal_context = nullptr;

extern "C" {
// Wrap host memory zero-copy when possible; newBufferWithBytesNoCopy returns
// nil for unaligned/oversized regions, in which case fall back to a real
// buffer with a memcpy in (and out for C).
static id<MTLBuffer> wrapOrCopy(id<MTLDevice> device, const void *ptr,
                                NSUInteger len, bool *copied) {
  id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:(void *)ptr
                                                length:len
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
  if (buf) {
    *copied = false;
    return buf;
  }
  *copied = true;
  return [device newBufferWithBytes:ptr
                             length:len
                            options:MTLResourceStorageModeShared];
}

void ternary_matmul(const int8_t *A, const int8_t *B_transposed, int32_t *C,
                    int M, int N, int K) {
  if (!g_metal_context) {
    g_metal_context = new MetalContext();
  }
  if (!g_metal_context->ok) {
    ternary_matmul_cpu(A, B_transposed, C, M, N, K);
    return;
  }

  bool cCopied = false, unused = false;
  id<MTLBuffer> bufferA = wrapOrCopy(g_metal_context->device, A,
                                     (M * K) * sizeof(int8_t), &unused);
  id<MTLBuffer> bufferB = wrapOrCopy(g_metal_context->device, B_transposed,
                                     (N * K) * sizeof(int8_t), &unused);
  // bufferC wraps the output array directly; after GPU execution the results
  // are already in C — no memcpy needed. If the wrap failed and we got a copy,
  // results are memcpy'd back below.
  id<MTLBuffer> bufferC = wrapOrCopy(g_metal_context->device, C,
                                     (M * N) * sizeof(int32_t), &cCopied);

  if (!bufferA || !bufferB || !bufferC) {
    ternary_matmul_cpu(A, B_transposed, C, M, N, K);
    return;
  }

  id<MTLCommandBuffer> commandBuffer =
      [g_metal_context->commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

  [encoder setComputePipelineState:g_metal_context->pipelineState];
  [encoder setBuffer:bufferA offset:0 atIndex:0];
  [encoder setBuffer:bufferB offset:0 atIndex:1];
  [encoder setBuffer:bufferC offset:0 atIndex:2];

  // Pass scalars
  [encoder setBytes:&M length:sizeof(int) atIndex:3];
  [encoder setBytes:&N length:sizeof(int) atIndex:4];
  [encoder setBytes:&K length:sizeof(int) atIndex:5];

  // Grid (M, N)
  // Threadgroup Size: 32x32 is standard good start for M1
  MTLSize gridSize = MTLSizeMake(N, M, 1);

  NSUInteger w = g_metal_context->pipelineState.threadExecutionWidth;
  NSUInteger h =
      g_metal_context->pipelineState.maxTotalThreadsPerThreadgroup / w;
  MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
  [encoder endEncoding];

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  // Zero-copy wrap writes results straight into C; a copied buffer needs an
  // explicit readback.
  if (cCopied) {
    memcpy(C, bufferC.contents, (M * N) * sizeof(int32_t));
  }
}
}
