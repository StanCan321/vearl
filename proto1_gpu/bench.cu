// VEARL - Prototype 1: GPU benchmark
// Measures what the consensus and the scheduler need: memory bandwidth,
// FP32 throughput, integer throughput ("hash-like" proxy), FP32 / FP16 tensor-core GEMM throughput.
// Build: make   |   Run: ./bench [--json]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)

// ---------- kernels ----------
__global__ void copy_k(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = in[i];
}

__global__ void fma_k(float* out, int iters) {
    float a = threadIdx.x * 1e-3f + 1.f, b = a + 0.1f, c = a + 0.2f, d = a + 0.3f;
    const float m = 1.0000001f, k = 0.9999999f;
    for (int i = 0; i < iters; i++) {
        a = fmaf(a, m, k); b = fmaf(b, m, k); c = fmaf(c, m, k); d = fmaf(d, m, k);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a + b + c + d;
}

// xorshift + rotate/add : ~9 32-bit integer operations per iteration
__global__ void int_k(uint32_t* out, int iters) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x + 1u, y = x * 2654435761u;
    for (int i = 0; i < iters; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        y = (y << 7 | y >> 25) + x;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = x ^ y;
}

__global__ void fill_f(float* p, size_t n, uint32_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
                 p[i] = ((h & 0xFFFF) / 65535.f - 0.5f) * 0.02f; }
}
__global__ void fill_h(__half* p, size_t n, uint32_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
                 p[i] = __float2half(((h & 0xFFFF) / 65535.f - 0.5f) * 0.02f); }
}

// ---------- timing helper ----------
struct Timer {
    cudaEvent_t a, b;
    Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    void start() { CK(cudaEventRecord(a)); }
    float stop_ms() { CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};

int main(int argc, char** argv) {
    bool json = argc > 1 && std::string_view(argv[1]) == "--json";
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int memClkKHz = 0, busBits = 0, smClkKHz = 0;
    CK(cudaDeviceGetAttribute(&memClkKHz, cudaDevAttrMemoryClockRate, 0));
    CK(cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, 0));
    CK(cudaDeviceGetAttribute(&smClkKHz, cudaDevAttrClockRate, 0));
    double peakBw = 2.0 * memClkKHz * 1e3 * (busBits / 8.0) / 1e9;
    Timer t;

    // 1) memory: copy 1 GiB -> 1 GiB (read + write)
    size_t bytes = 1ull << 30, n4 = bytes / sizeof(float4);
    float4 *in, *out; CK(cudaMalloc(&in, bytes)); CK(cudaMalloc(&out, bytes));
    CK(cudaMemset(in, 1, bytes));
    copy_k<<<p.multiProcessorCount * 8, 256>>>(in, out, n4); CK(cudaDeviceSynchronize());
    t.start(); for (int i = 0; i < 20; i++) copy_k<<<p.multiProcessorCount * 8, 256>>>(in, out, n4);
    double bw = 20.0 * 2 * bytes / (t.stop_ms() / 1e3) / 1e9;
    CK(cudaFree(in)); CK(cudaFree(out));

    // 2) FP32 FMA
    int blocks = p.multiProcessorCount * 8, thr = 256, iters = 1 << 15;
    float* fo; CK(cudaMalloc(&fo, blocks * thr * sizeof(float)));
    fma_k<<<blocks, thr>>>(fo, 1024); CK(cudaDeviceSynchronize());
    t.start(); fma_k<<<blocks, thr>>>(fo, iters);
    double tflops = (double)blocks * thr * iters * 4 * 2 / (t.stop_ms() / 1e3) / 1e12;
    CK(cudaFree(fo));

    // 3) integer
    uint32_t* io; CK(cudaMalloc(&io, blocks * thr * sizeof(uint32_t)));
    int_k<<<blocks, thr>>>(io, 1024); CK(cudaDeviceSynchronize());
    t.start(); int_k<<<blocks, thr>>>(io, iters);
    double giops = (double)blocks * thr * iters * 9 / (t.stop_ms() / 1e3) / 1e9;
    CK(cudaFree(io));

    // 4) GEMM (cuBLAS) n x n
    const int N = 8192; size_t ne = (size_t)N * N;
    cublasHandle_t h; CB(cublasCreate(&h));
    float *A, *B, *C; CK(cudaMalloc(&A, ne * 4)); CK(cudaMalloc(&B, ne * 4)); CK(cudaMalloc(&C, ne * 4));
    fill_f<<<(ne + 255) / 256, 256>>>(A, ne, 1); fill_f<<<(ne + 255) / 256, 256>>>(B, ne, 2);
    const float one = 1.f, zero = 0.f;
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, A, N, B, N, &zero, C, N)); CK(cudaDeviceSynchronize());
    t.start(); for (int i = 0; i < 5; i++) CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, A, N, B, N, &zero, C, N));
    double sgemm = 5.0 * 2.0 * N * (double)N * N / (t.stop_ms() / 1e3) / 1e12;
    CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));

    __half *Ah, *Bh, *Ch; CK(cudaMalloc(&Ah, ne * 2)); CK(cudaMalloc(&Bh, ne * 2)); CK(cudaMalloc(&Ch, ne * 2));
    fill_h<<<(ne + 255) / 256, 256>>>(Ah, ne, 3); fill_h<<<(ne + 255) / 256, 256>>>(Bh, ne, 4);
    auto hgemm = [&]() { CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, Ah, CUDA_R_16F, N, Bh, CUDA_R_16F, N,
                                          &zero, Ch, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); };
    hgemm(); CK(cudaDeviceSynchronize());
    t.start(); for (int i = 0; i < 10; i++) hgemm();
    double hgemm_tf = 10.0 * 2.0 * N * (double)N * N / (t.stop_ms() / 1e3) / 1e12;
    CK(cudaFree(Ah)); CK(cudaFree(Bh)); CK(cudaFree(Ch)); CB(cublasDestroy(h));

    if (json) {
        printf("{\"name\":\"%s\",\"sm_count\":%d,\"compute_capability\":\"%d.%d\",\"vram_gib\":%.1f,"
               "\"mem_bw_peak_gbs\":%.0f,\"mem_bw_copy_gbs\":%.0f,\"fp32_fma_tflops\":%.2f,"
               "\"int32_giops\":%.0f,\"sgemm_fp32_tflops\":%.2f,\"gemm_fp16_tensor_tflops\":%.2f}\n",
               p.name, p.multiProcessorCount, p.major, p.minor, p.totalGlobalMem / 1073741824.0,
               peakBw, bw, tflops, giops, sgemm, hgemm_tf);
    } else {
        printf("GPU               : %s (SM %d.%d, %d SMs, %.1f GiB)\n", p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0);
        printf("Memory            : %.0f GB/s measured / %.0f GB/s theoretical (%.0f %%)\n", bw, peakBw, 100 * bw / peakBw);
        printf("FP32 FMA          : %.2f TFLOPS\n", tflops);
        printf("INT32 (xorshift)  : %.0f Gops/s\n", giops);
        printf("SGEMM FP32 8192^3 : %.2f TFLOPS\n", sgemm);
        printf("GEMM FP16 tensor  : %.2f TFLOPS\n", hgemm_tf);
    }
    return 0;
}
