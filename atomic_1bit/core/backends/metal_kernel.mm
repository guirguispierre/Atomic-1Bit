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

    // Try current directory first
    // In Python assumption: os.getcwd() usually project root.
    // But lib is in core/.
    // Let's use specific path for now or "default.metallib".

    // Logic: Try to create library from file "default.metallib" in
    // "atomic_1bit/core" Adjust path logic for real deployment. HACK: Hardcoded
    // relative path for dev environment.
    NSString *libPath = @"atomic_1bit/core/default.metallib";
    NSURL *libOnDisk = [NSURL fileURLWithPath:libPath];

    id<MTLLibrary> defaultLibrary = [device newLibraryWithURL:libOnDisk
                                                        error:&error];
    if (!defaultLibrary) {
      // Fallback: try just "default.metallib" if running from core/
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

  // Create buffers
  // We use NoCopy to wrap the existing numpy memory.
  // Note: The memory MUST be page-aligned for best results, but Managed mode
  // might be strict. Shared mode is fine for Apple Silicon. Python
  // bytearray/numpy might not be page aligned. If it fails, we copy. Safe path:
  // use Shared mode with copy if needed, but lets try NoCopy.
  // 'newBufferWithBytesNoCopy' requires pointer to be page aligned usually?
  // Actually on macOS with Shared mode it handles it, but alignment is
  // performance critical.
  //
  // SAFE IMPLEMENTATION: newBufferWithBytes (copying) to guarantee correctness
  // first. Optimization: Zero-copy can be added later if we control allocation
  // in Python.

  // Let's use 'newBufferWithBytes' (Copy) for V1.2 robustness.
  // Bandwidth is high on M-series, but copy is overhead.

  id<MTLBuffer> bufferA =
      [g_metal_context->device newBufferWithBytes:A
                                           length:(M * K) * sizeof(int8_t)
                                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferB =
      [g_metal_context->device newBufferWithBytes:B_transposed
                                           length:(N * K) * sizeof(int8_t)
                                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferC =
      [g_metal_context->device newBufferWithBytes:C
                                           length:(M * N) * sizeof(int32_t)
                                          options:MTLResourceStorageModeShared];

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

  // Copy back result C
  memcpy(C, [bufferC contents], (M * N) * sizeof(int32_t));
}
}
