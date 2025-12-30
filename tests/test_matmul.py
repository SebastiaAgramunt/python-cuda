import numpy as np
import matmul


def test_tiled_and_cublas_agree():
    M, K, N = 128, 64, 256

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    C_tiled = matmul.matmul(A, B)
    C_blas = matmul.matmul_cublas(A, B)

    assert C_tiled.shape == (M, N)
    assert C_blas.shape == (M, N)

    np.testing.assert_allclose(C_tiled, C_blas, rtol=1e-4, atol=1e-4)
