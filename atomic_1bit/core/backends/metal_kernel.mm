#include "../kernel.h"
#include <Metal/Metal.h>
#include <iostream>

// Singleton for Metal Context
struct MetalContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> pipelineState;

  MetalContext() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::cerr << "Error: No Metal device found." << std::endl;
      exit(1);
    }
    commandQueue = [device newCommandQueue];

    // Load default library
    NSError *error = nil;

    // The metallib is expected to be next to the dylib.
    // But for simplicity in this build, we might load from default library if
    // embedded or from file. Since we compile to 'default.metallib', we load
    // it.

    // We need to find the path to default.metallib relative to where we are
    // running, OR we can assume it's in the current directory or adjacent. For
    // robustness, let's try to load from the main bundle or current directory.

    // Resolve metallib path relative to this source file's directory.
    // __FILE__ expands to the full path of this .mm file at compile time:
    //   .../atomic_1bit/core/backends/metal_kernel.mm
    // Deleting the last path component ("backends") gives us the "core/"
    // directory, which is where default.metallib is placed by the Makefile.
    NSString *sourceDir =
        [[[NSString stringWithUTF8String:__FILE__]
            stringByDeletingLastPathComponent]   // strip "metal_kernel.mm"
            stringByDeletingLastPathComponent];  // strip "backends/"
    NSString *libPath =
        [sourceDir stringByAppendingPathComponent:@"default.metallib"];
    NSURL *libOnDisk = [NSURL fileURLWithPath:libPath];

    id<MTLLibrary> defaultLibrary = [device newLibraryWithURL:libOnDisk
                                                        error:&error];
    if (!defaultLibrary) {
      // Fallback: try "default.metallib" relative to cwd (e.g. when running
      // from atomic_1bit/core/ directly during development).
      libOnDisk = [NSURL fileURLWithPath:@"default.metallib"];
      defaultLibrary = [device newLibraryWithURL:libOnDisk error:&error];
    }

    if (!defaultLibrary) {
      std::cerr << "Error: Could not load default.metallib: " <<
          [[error localizedDescription] UTF8String] << std::endl;
      exit(1);
    }

    id<MTLFunction> kernelFunction =
        [defaultLibrary newFunctionWithName:@"ternary_matmul_shader"];
    if (!kernelFunction) {
      std::cerr
          << "Error: Could not find function 'ternary_matmul_shader' in library"
          << std::endl;
      exit(1);
    }

    pipelineState = [device newComputePipelineStateWithFunction:kernelFunction
                                                          error:&error];
    if (!pipelineState) {
      std::cerr << "Error: Could not create pipeline state: " <<
          [[error localizedDescription] UTF8String] << std::endl;
      exit(1);
    }
  }
};

static MetalContext *g_metal_context = nullptr;

extern "C" {
void ternary_matmul(const int8_t *A, const int8_t *B_transposed, int32_t *C,
                    int M, int N, int K) {
  if (!g_metal_context) {
    g_metal_context = new MetalContext();
  }

  // Zero-copy buffers: wrap the caller's memory directly using
  // newBufferWithBytesNoCopy with MTLResourceStorageModeShared.
  // This is safe on M-series (Apple Silicon) unified memory because the CPU
  // and GPU share the same physical memory — no explicit DMA transfer occurs
  // and no staging copy is needed. The nil deallocator means Metal will NOT
  // free the underlying memory when the buffer is released; the caller (Python
  // / ctypes) owns and manages the lifetime of A, B_transposed, and C.
  // Note: on Apple Silicon, MTLResourceStorageModeShared does not require
  // page alignment for correctness, though aligned allocations are faster.
  id<MTLBuffer> bufferA = [g_metal_context->device
      newBufferWithBytesNoCopy:(void *)A
                        length:(M * K) * sizeof(int8_t)
                       options:MTLResourceStorageModeShared
                   deallocator:nil];
  id<MTLBuffer> bufferB = [g_metal_context->device
      newBufferWithBytesNoCopy:(void *)B_transposed
                        length:(N * K) * sizeof(int8_t)
                       options:MTLResourceStorageModeShared
                   deallocator:nil];
  // bufferC wraps the output array directly; after GPU execution the results
  // are already in C — no memcpy needed.
  id<MTLBuffer> bufferC = [g_metal_context->device
      newBufferWithBytesNoCopy:(void *)C
                        length:(M * N) * sizeof(int32_t)
                       options:MTLResourceStorageModeShared
                   deallocator:nil];

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

  // No memcpy needed: bufferC is a zero-copy wrapper around C, so the GPU
  // has already written the results directly into the caller's array.
}
}
