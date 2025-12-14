import numpy as np
import time
from wrapper import ternary_matmul

def run_verification(M, N, K, verbose=False):
    print(f"\n--- Verification: M={M}, N={N}, K={K} ---")
    
    # 1. Generate Dummy Data
    # Activations: Random INT8
    A = np.random.randint(-127, 128, size=(M, K), dtype=np.int8)
    
    # Weights: Random ternary {-1, 0, 1}
    raw = np.random.randint(0, 3, size=(K, N), dtype=np.int8)
    B = raw - 1
    
    # 2. Run Atomic Kernel
    try:
        start_t = time.time()
        C_atomic = ternary_matmul(A, B)
        end_t = time.time()
        print(f"Atomic Kernel Time: {(end_t - start_t)*1000:.4f} ms")
    except Exception as e:
        print(f"Kernel Failed: {e}")
        return False

    # 3. Verification (Python INT8 reference with INT32 accum)
    start_ref = time.time()
    C_ref = np.dot(A.astype(np.int32), B.astype(np.int32))
    end_ref = time.time()
    print(f"Reference Numpy Time: {(end_ref - start_ref)*1000:.4f} ms")
    
    # Debug Prints
    if verbose:
        print("\n[DEBUG] First Row Inputs A[0, :8]:", A[0, :8])
        print("[DEBUG] First Col Weights B[:8, 0]:", B[:8, 0])
        print("[DEBUG] Reference Output C_ref[0, :8]:", C_ref[0, :8])
        print("[DEBUG] Kernel Output C_atomic[0, :8]:", C_atomic[0, :8])

    # Check
    if np.array_equal(C_atomic, C_ref):
        print(">> SUCCESS: Kernel Output Matches Reference.")
        return True
    else:
        print(">> FAILURE: Mismatch detected!")
        diff = np.abs(C_atomic - C_ref)
        print(f"Max diff: {np.max(diff)}")
        print(f"Mean diff: {np.mean(diff)}")
        return False

def run_demo():
    # 1. Tiny Sanity Check
    if not run_verification(M=1, N=8, K=16, verbose=True):
        print("\n!!! Tiny Verification Failed - Skipping Large Test !!!")
        return

    # 2. Large Scale
    print("\n--- Running Large Scale System Check ---")
    run_verification(M=1, N=4096, K=4096)

if __name__ == "__main__":
    run_demo()
