// src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>
#include <stdexcept>
#include "tiledMultiply.h"
#include "cuBLASMultiply.h"  // <-- add this

namespace py = pybind11;

namespace {

inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err)
        );
    }
}

// -----------------------
// Tiled kernel version
// -----------------------
py::array_t<float> matmul_tiled(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B)
{
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }

    const auto M  = static_cast<std::size_t>(A.shape(0));
    const auto K  = static_cast<std::size_t>(A.shape(1));
    const auto Kb = static_cast<std::size_t>(B.shape(0));
    const auto N  = static_cast<std::size_t>(B.shape(1));

    if (K != Kb) {
        throw std::runtime_error("Inner dimensions must match: A(M,K) @ B(K,N)");
    }

    py::array_t<float> C({static_cast<py::ssize_t>(M),
                          static_cast<py::ssize_t>(N)});

    const float* hA = A.data();
    const float* hB = B.data();
    float* hC       = C.mutable_data();

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    std::size_t bytesA = M * K * sizeof(float);
    std::size_t bytesB = K * N * sizeof(float);
    std::size_t bytesC = M * N * sizeof(float);

    cuda_check(cudaMalloc(&dA, bytesA), "cudaMalloc dA failed");
    cuda_check(cudaMalloc(&dB, bytesB), "cudaMalloc dB failed");
    cuda_check(cudaMalloc(&dC, bytesC), "cudaMalloc dC failed");

    cuda_check(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice),
               "cudaMemcpy A failed");
    cuda_check(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice),
               "cudaMemcpy B failed");

    tiledMultiply_call(dA, dB, dC, M, K, N);

    cuda_check(cudaGetLastError(), "Kernel launch failed");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    cuda_check(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost),
               "cudaMemcpy C failed");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return C;
}

// -----------------------
// cuBLAS version
// -----------------------
py::array_t<float> matmul_cublas(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B)
{
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }

    const auto M  = static_cast<std::size_t>(A.shape(0));
    const auto K  = static_cast<std::size_t>(A.shape(1));
    const auto Kb = static_cast<std::size_t>(B.shape(0));
    const auto N  = static_cast<std::size_t>(B.shape(1));

    if (K != Kb) {
        throw std::runtime_error("Inner dimensions must match: A(M,K) @ B(K,N)");
    }

    py::array_t<float> C({static_cast<py::ssize_t>(M),
                          static_cast<py::ssize_t>(N)});

    const float* hA = A.data();
    const float* hB = B.data();
    float* hC       = C.mutable_data();

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    std::size_t bytesA = M * K * sizeof(float);
    std::size_t bytesB = K * N * sizeof(float);
    std::size_t bytesC = M * N * sizeof(float);

    cuda_check(cudaMalloc(&dA, bytesA), "cudaMalloc dA failed");
    cuda_check(cudaMalloc(&dB, bytesB), "cudaMalloc dB failed");
    cuda_check(cudaMalloc(&dC, bytesC), "cudaMalloc dC failed");

    cuda_check(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice),
               "cudaMemcpy A failed");
    cuda_check(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice),
               "cudaMemcpy B failed");

    // Stream variable passed to your cuBLAS wrapper.
    // Your implementation creates/sets the stream internally.
    cudaStream_t stream = nullptr;
    cuBLASmultiply_call(dA, dB, dC, M, K, N, stream);

    cuda_check(cudaGetLastError(), "cuBLAS call failed");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    cuda_check(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost),
               "cudaMemcpy C failed");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return C;
}

} // namespace

PYBIND11_MODULE(matmul, m) {
    m.doc() = "Tiled and cuBLAS CUDA matrix multiplication (A[M,K] @ B[K,N])";

    m.def("matmul",
          &matmul_tiled,
          py::arg("A"),
          py::arg("B"),
          R"doc(
Multiply two matrices using a custom CUDA tiled kernel.

Parameters
----------
A : numpy.ndarray (M, K), float32
B : numpy.ndarray (K, N), float32

Returns
-------
C : numpy.ndarray (M, N), float32
)doc");

    m.def("matmul_cublas",
          &matmul_cublas,
          py::arg("A"),
          py::arg("B"),
          R"doc(
Multiply two matrices using cuBLAS (cublasSgemm) on the GPU.

Parameters
----------
A : numpy.ndarray (M, K), float32
B : numpy.ndarray (K, N), float32

Returns
-------
C : numpy.ndarray (M, N), float32
)doc");
}
