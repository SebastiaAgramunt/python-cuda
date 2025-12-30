import numpy as np
import matmul

A = np.random.rand(128, 64).astype(np.float32)
B = np.random.rand(64, 256).astype(np.float32)

C_tiled = matmul.matmul(A, B)
C_blas  = matmul.matmul_cublas(A, B)

np.testing.assert_allclose(C_tiled, C_blas, rtol=1e-4, atol=1e-4)


# print C_tiled
print("C_tiled:")
print(C_tiled)

# print C_blas
print("C_blas:")
print(C_blas)