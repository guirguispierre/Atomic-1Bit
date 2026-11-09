import ctypes
import os
import subprocess
import numpy as np

# Load the shared library
# Construct absolute path to ensure it loads correctly
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.normpath(os.path.join(_CUR_DIR, "../core"))
_LIB_PATH = os.path.join(_CORE_DIR, "libatomic.so")


def _try_build_libatomic():
    """Attempt to build libatomic.so via the core Makefile.

    Returns None on success, or a string describing the failure so the
    caller can fold it into the FileNotFoundError below. Never raises:
    a missing toolchain must not turn the import into a subprocess crash.
    """
    try:
        proc = subprocess.run(
            ["make", "-C", _CORE_DIR],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except OSError as e:
        return f"could not run 'make -C {_CORE_DIR}' ({e})"
    except subprocess.SubprocessError as e:
        return f"'make -C {_CORE_DIR}' did not complete ({e})"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        return f"'make -C {_CORE_DIR}' exited with code {proc.returncode}: {tail}"
    if not os.path.exists(_LIB_PATH):
        return f"'make -C {_CORE_DIR}' succeeded but {_LIB_PATH} is still missing"
    return None


if not os.path.exists(_LIB_PATH):
    build_error = _try_build_libatomic()
    if build_error is not None:
        raise FileNotFoundError(
            f"Could not find shared library at {_LIB_PATH}, and the automatic "
            f"build failed: {build_error}.\n"
            f"Build it manually with 'make -C {_CORE_DIR}' "
            f"(requires make and a C++17 compiler)."
        )
    print(f"[atomic_1bit] Built {_LIB_PATH} (make -C {_CORE_DIR})")

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
    
    # KERNEL FIX: The C++ kernel expects B to be (N, K) layout (Transposed).
    # We transpose here so the user can pass standard (K, N) weights.
    # Standard: (K, N). Transposed: (N, K).
    # We make it contiguous row-major (N, K) so kernel reads rows efficiently.
    B_transposed = weights.T
    B = np.ascontiguousarray(B_transposed, dtype=np.int8)
    
    M, K = A.shape
    N, K_w = B.shape # B is (N, K) now
    
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
