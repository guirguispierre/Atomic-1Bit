import numpy as np
import time
from wrapper import ternary_matmul

def run_demo():
    print("--- Atomic-1Bit: System Check ---")
    
    # Dimensions
    M = 1    # Batch size (1 token)
    N = 4096 # Output features (Hidden size)
    K = 4096 # Input features
    
    print(f"Config: M={M}, N={N}, K={K}")
    
    # 1. Generate Dummy Data
    # Activations: Random INT8
    A = np.random.randint(-128, 127, size=(M, K), dtype=np.int8)
    
    # Weights: Random ternary {-1, 0, 1}
    # Create random indices 0, 1, 2 and map to -1, 0, 1
    raw = np.random.randint(0, 3, size=(K, N), dtype=np.int8)
    B = raw - 1
    
    print("Data generated.")
    
    # 2. Run Atomic Kernel
    start_t = time.time()
    C_atomic = ternary_matmul(A, B)
    end_t = time.time()
    
    print(f"Atomic Kernel Time: {(end_t - start_t)*1000:.4f} ms")
    
    # 3. Verification (Python INT8 reference)
    # WARNING: This will be slow for large matrices in pure Python/Numpy comparison if not careful, 
    # but Numpy dot is optimized.
    # Note: Numpy dot might do full int64 mul, but result should match.
    print("Verifying correctness...")
    start_ref = time.time()
    C_ref = np.dot(A.astype(np.int32), B.astype(np.int32))
    end_ref = time.time()
    print(f"Reference Numpy Time: {(end_ref - start_ref)*1000:.4f} ms")
    
    # Check
    if np.array_equal(C_atomic, C_ref):
        print(">> SUCCESS: Kernel Output Matches Reference.")
    else:
        print(">> FAILURE: Mismatch detected!")
        # Debug info
        diff = np.abs(C_atomic - C_ref)
        print(f"Max diff: {np.max(diff)}")
        print(f"Mean diff: {np.mean(diff)}")

if __name__ == "__main__":
    run_demo()
