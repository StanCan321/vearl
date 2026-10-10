// int8 ceiling of the A10 measured with cuBLAS (int8 -> int32 GEMM on tensor cores), to place the VEARL kernel.
// Build: nvcc -O3 -arch=sm_86 int8_ceiling.cu -lcublas -o int8_ceiling
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    for (int n : {4096, 8192}) {
        size_t nn = (size_t)n * n;
        int8_t *A, *B; int32_t* C;
        cudaMalloc(&A, nn); cudaMalloc(&B, nn); cudaMalloc(&C, nn * 4);
        cudaMemset(A, 1, nn); cudaMemset(B, 1, nn);
        cublasHandle_t h; cublasCreate(&h);
        int32_t alpha = 1, beta = 0;
        auto run = [&]() {
            // C = A^T * B in int8: the cuBLAS int8 GEMMs require op(A)=T, op(B)=N on the tensor cores
            return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, A, CUDA_R_8I, n, B, CUDA_R_8I, n, &beta, C,
                                CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };
        if (run() != CUBLAS_STATUS_SUCCESS) { printf("n=%d: GemmEx int8 refused\n", n); continue; }
        cudaDeviceSynchronize();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int i = 0; i < 10; i++) run();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 10;
        printf("n=%d: cuBLAS int8 %.2f ms -> %.1f TOPS\n", n, ms, 2.0 * n * (double)n * n / (ms * 1e-3) / 1e12);
        cudaFree(A); cudaFree(B); cudaFree(C); cublasDestroy(h);
    }
    return 0;
}
