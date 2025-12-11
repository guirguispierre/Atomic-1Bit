import ctypes
import os
import numpy as np

# Load the shared library
# Construct absolute path to ensure it loads correctly
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_CUR_DIR, "../core/libatomic.so")

if not os.path.exists(_LIB_PATH):
    raise FileNotFoundError(f"Could not find shared library at {_LIB_PATH}. Did you run 'make' in core/?")

_lib = ctypes.CDLL(_LIB_PATH)

# void ternary_matmul(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K);
_lib.ternary_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_int8),  # A
    ctypes.POINTER(ctypes.c_int8),  # B
    ctypes.POINTER(ctypes.c_int32), # C
    ctypes.c_int, # M
    ctypes.c_int, # N
    ctypes.c_int  # K
]
_lib.ternary_matmul.restype = None

def ternary_matmul(activations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Executes the Atomic-1Bit Ternary Matmul Kernel.
    
    Args:
        activations: INT8 numpy array of shape (M, K)
        weights: INT8 numpy array of shape (K, N) with values {-1, 0, 1}
        
    Returns:
        INT32 numpy array of shape (M, N)
    """
    # Ensure inputs are contiguous C-style arrays of correct type
    A = np.ascontiguousarray(activations, dtype=np.int8)
    B = np.ascontiguousarray(weights, dtype=np.int8)
    
    M, K = A.shape
    K_w, N = B.shape
    
    if K != K_w:
        raise ValueError(f"Shape mismatch: A({M},{K}) vs B({K_w},{N})")
    
    # Pre-allocate output
    C = np.zeros((M, N), dtype=np.int32)
    
    # Get pointers
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    
    # Run Nucleus
    _lib.ternary_matmul(A_ptr, B_ptr, C_ptr, M, N, K)
    
    return C
