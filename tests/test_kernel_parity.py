import pytest
import numpy as np
import os


def _kernel_available():
    """Check if the C++ kernel (libatomic.so) is built and available."""
    lib_path = os.path.join(
        os.path.dirname(__file__), "..", "atomic_1bit", "core", "libatomic.so"
    )
    return os.path.exists(lib_path)


@pytest.mark.skipif(not _kernel_available(), reason="C++ kernel not built (run 'make' in atomic_1bit/core/)")
class TestKernelParity:
    """Critical parity tests: C++ kernel must produce bit-exact results vs NumPy reference."""

    @pytest.fixture(autouse=True)
    def _import_kernel(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "atomic_1bit", "python"))
        from wrapper import ternary_matmul
        self.ternary_matmul = ternary_matmul

    def _verify(self, M, N, K):
        """Run parity check for given dimensions."""
        # Random INT8 activations
        A = np.random.randint(-127, 128, size=(M, K), dtype=np.int8)
        # Random ternary weights {-1, 0, 1}
        B = np.random.randint(0, 3, size=(K, N), dtype=np.int8) - 1

        # C++ kernel
        C_kernel = self.ternary_matmul(A, B)

        # NumPy reference (INT32 accumulation)
        C_ref = np.dot(A.astype(np.int32), B.astype(np.int32))

        np.testing.assert_array_equal(
            C_kernel, C_ref,
            err_msg=f"Parity FAILED for M={M}, N={N}, K={K}"
        )

    def test_tiny_parity(self):
        """1x8x16: Minimal sanity check."""
        self._verify(M=1, N=8, K=16)

    def test_single_row(self):
        """1x64x64: Single-row matmul."""
        self._verify(M=1, N=64, K=64)

    def test_square(self):
        """32x32x32: Square matrix."""
        self._verify(M=32, N=32, K=32)

    def test_rectangular(self):
        """4x128x64: Rectangular matmul."""
        self._verify(M=4, N=128, K=64)

    def test_flagship_dims(self):
        """1x320x320: Flagship model single-token dimensions."""
        self._verify(M=1, N=320, K=320)

    def test_large_scale(self):
        """1x4096x4096: Large-scale stress test."""
        self._verify(M=1, N=4096, K=4096)

    def test_all_zeros_weights(self):
        """All-zero weights should produce all-zero output."""
        A = np.random.randint(-127, 128, size=(2, 32), dtype=np.int8)
        B = np.zeros((32, 16), dtype=np.int8)
        C = self.ternary_matmul(A, B)
        np.testing.assert_array_equal(C, np.zeros((2, 16), dtype=np.int32))

    def test_all_ones_weights(self):
        """All-ones weights: output should equal row sums of activation."""
        A = np.random.randint(-127, 128, size=(2, 8), dtype=np.int8)
        B = np.ones((8, 4), dtype=np.int8)
        C = self.ternary_matmul(A, B)
        expected = A.astype(np.int32).sum(axis=1, keepdims=True) * np.ones((1, 4), dtype=np.int32)
        np.testing.assert_array_equal(C, expected)

    def test_all_neg_ones_weights(self):
        """All -1 weights: output should be negated row sums."""
        A = np.random.randint(-127, 128, size=(2, 8), dtype=np.int8)
        B = -np.ones((8, 4), dtype=np.int8)
        C = self.ternary_matmul(A, B)
        expected = -A.astype(np.int32).sum(axis=1, keepdims=True) * np.ones((1, 4), dtype=np.int32)
        np.testing.assert_array_equal(C, expected)

    def test_int8_min_activation(self):
        # guirguispierre 2026-08-10 - -128 against -1 is +128, which int8 cannot
        # hold; K=64 keeps this on the vectorized path
        A = np.full((1, 64), -128, dtype=np.int8)
        B = np.full((64, 8), -1, dtype=np.int8)
        C = self.ternary_matmul(A, B)
        expected = A.astype(np.int32) @ B.astype(np.int32)
        np.testing.assert_array_equal(C, expected)
        assert C[0, 0] == 8192, "int8 overflow in the widening multiply"

    def test_full_int8_activation_range(self):
        rng = np.random.default_rng(0)
        for _ in range(40):
            K = int(rng.integers(1, 300))
            N = int(rng.integers(1, 64))
            A = rng.integers(-128, 128, size=(2, K)).astype(np.int8)
            B = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
            np.testing.assert_array_equal(
                self.ternary_matmul(A, B),
                A.astype(np.int32) @ B.astype(np.int32),
                err_msg=f"mismatch at K={K}, N={N}",
            )
