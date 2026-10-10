// VEARL - GPU kernel of the NoisyGEMM-T work (spec v2, see src/work.rs).
// - int8 tensor cores (WMMA 16x16x16), one 16x16 output tile per warp
// - folding of the transcript every fold_k steps of k (XOR + rotation), then single-chunk blake3
// - result bit-for-bit identical to the CPU reference (checked by the Rust test `gpu_matches_cpu`)
// C API: vearl_gpu_new / vearl_gpu_run / vearl_gpu_free
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return -1; } } while (0)

// ---------------- blake3 (single chunk, plain hash mode) ----------------
__constant__ uint32_t B3_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                  0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
__constant__ uint8_t B3_PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
// header of the attempt: "vearl/jackpot/v2" (16 B) || sigma (32 B), one per sigma of the batch (global memory)

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ __forceinline__ void g(uint32_t* s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    s[a] = s[a] + s[b] + mx; s[d] = rotr32(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + my; s[d] = rotr32(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 7);
}

// permutation of the message words between two rounds, WRITTEN OUT (constant indices): with an array in constant
// memory the indexing was dynamic and forced the message words into local memory.
__device__ __forceinline__ void b3_permute(uint32_t m[16]) {
    uint32_t t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
    #pragma unroll
    for (int i = 0; i < 16; i++) m[i] = t[i];
}

__device__ void b3_compress(uint32_t cv[8], const uint32_t block[16], uint32_t block_len, uint32_t flags) {
    uint32_t s[16], m[16];
    for (int i = 0; i < 8; i++) s[i] = cv[i];
    for (int i = 0; i < 4; i++) s[8 + i] = B3_IV[i];
    s[12] = 0; s[13] = 0; s[14] = block_len; s[15] = flags;
    for (int i = 0; i < 16; i++) m[i] = block[i];
    #pragma unroll
    for (int r = 0; r < 7; r++) {
        g(s, 0, 4, 8, 12, m[0], m[1]);  g(s, 1, 5, 9, 13, m[2], m[3]);
        g(s, 2, 6, 10, 14, m[4], m[5]); g(s, 3, 7, 11, 15, m[6], m[7]);
        g(s, 0, 5, 10, 15, m[8], m[9]);  g(s, 1, 6, 11, 12, m[10], m[11]);
        g(s, 2, 7, 8, 13, m[12], m[13]); g(s, 3, 4, 9, 14, m[14], m[15]);
        if (r < 6) b3_permute(m);
    }
    for (int i = 0; i < 8; i++) cv[i] = s[i] ^ s[i + 8];
}

// blake3: full compression (16-word output) with arbitrary counter and key. Used (1) to derive the key of the context
// "vearl/noise/v1" (derive_key mode) and (2) to generate the noise XOF stream: output block j = compress(key, sigma, counter j).
__device__ void b3_compress_xof(const uint32_t key[8], const uint32_t block[16], uint32_t counter, uint32_t block_len, uint32_t flags, uint32_t out[16]) {
    uint32_t s[16], m[16];
    for (int i = 0; i < 8; i++) s[i] = key[i];
    for (int i = 0; i < 4; i++) s[8 + i] = B3_IV[i];
    s[12] = counter; s[13] = 0; s[14] = block_len; s[15] = flags;
    for (int i = 0; i < 16; i++) m[i] = block[i];
    #pragma unroll
    for (int r = 0; r < 7; r++) {
        g(s, 0, 4, 8, 12, m[0], m[1]);  g(s, 1, 5, 9, 13, m[2], m[3]);
        g(s, 2, 6, 10, 14, m[4], m[5]); g(s, 3, 7, 11, 15, m[6], m[7]);
        g(s, 0, 5, 10, 15, m[8], m[9]);  g(s, 1, 6, 11, 12, m[10], m[11]);
        g(s, 2, 7, 8, 13, m[12], m[13]); g(s, 3, 4, 9, 14, m[14], m[15]);
        if (r < 6) b3_permute(m);
    }
    for (int i = 0; i < 8; i++) { out[i] = s[i] ^ s[i + 8]; out[i + 8] = s[i + 8] ^ key[i]; }
}

__device__ uint32_t g_noise_key[8]; // = blake3::derive_key_context("vearl/noise/v1")

// A single thread: hash of the context (14 bytes) with the IV key, flags CHUNK_START|CHUNK_END|ROOT|DERIVE_KEY_CONTEXT.
__global__ void noise_key_kernel() {
    const char ctx[15] = "vearl/noise/v1";
    uint32_t blk[16];
    for (int i = 0; i < 16; i++) blk[i] = 0;
    for (int i = 0; i < 14; i++) blk[i >> 2] |= ((uint32_t)(uint8_t)ctx[i]) << (8 * (i & 3));
    uint32_t iv[8], out[16];
    for (int i = 0; i < 8; i++) iv[i] = B3_IV[i];
    b3_compress_xof(iv, blk, 0, 14, 1 | 2 | 8 | 32, out);
    for (int i = 0; i < 8; i++) g_noise_key[i] = out[i];
}

// Noise of the whole batch: the XOF stream of sigma (derive_key mode: message = sigma on 32 bytes, flags
// CHUNK_START|CHUNK_END|ROOT|DERIVE_KEY_MATERIAL) is cut into 4 consecutive buffers [EL | ER | FL | FR] of nr bytes;
// each byte b is worth (b % 3) - 1. One thread = one block of 64 bytes. nr = n * rank is a multiple of 64 (n multiple of 128).
// hdrs: 12 words per sigma, sigma = words 4..11.
__global__ void noise_gen(const uint32_t* __restrict__ hdrs, int8_t* el, int8_t* er, int8_t* fl, int8_t* fr, size_t nr) {
    const size_t per = nr / 64;                        // blocks of 64 bytes per buffer
    const size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (j >= 4 * per) return;
    const size_t sg = blockIdx.y;
    uint32_t blk[16], out[16], key[8];
    for (int i = 0; i < 8; i++) { blk[i] = hdrs[sg * 12 + 4 + i]; blk[8 + i] = 0; key[i] = g_noise_key[i]; }
    b3_compress_xof(key, blk, (uint32_t)j, 32, 1 | 2 | 8 | 64, out);
    const size_t which = j / per, off = (j % per) * 64;
    int8_t* dst = (which == 0 ? el : which == 1 ? er : which == 2 ? fl : fr) + sg * nr + off;
    // 64 bytes -> 16 packed words (b % 3 - 1 on each byte), written in 4 stores of 16 bytes
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        uint32_t pk[4];
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            uint32_t v = out[q * 4 + w], acc = 0;
            #pragma unroll
            for (int b = 0; b < 4; b++) acc |= (uint32_t)(((int)((v >> (8 * b)) & 0xff) % 3 - 1) & 0xff) << (8 * b);
            pk[w] = acc;
        }
        d4[q] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
}

// message = hdr(12 words) || ti || tj || fold(64 words) = 78 words = 312 bytes = 5 blocks
__device__ void jackpot_hash(const uint32_t* hdr, uint32_t ti, uint32_t tj, const uint32_t* fold, uint32_t out[8]) {
    uint32_t msg[80];
    for (int i = 0; i < 12; i++) msg[i] = hdr[i];
    msg[12] = ti; msg[13] = tj;
    for (int i = 0; i < 64; i++) msg[14 + i] = fold[i];
    msg[78] = 0; msg[79] = 0;
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    for (int b = 0; b < 5; b++) {
        uint32_t flags = (b == 0 ? 1u : 0u) | (b == 4 ? (2u | 8u) : 0u); // CHUNK_START | CHUNK_END | ROOT
        b3_compress(cv, msg + 16 * b, b == 4 ? 56u : 64u, flags);
    }
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// ---------------- noise on TENSOR CORES: out = x + L*R (n x n, depth rank <= 64) ----------------
// A' = A + EL*ER  and  B'^T = B^T + FR^T * FL^T are GEMMs of depth `rank`: they are done with int8 mma
// instead of scalar loops. The operands are first "packed" into 16x16 blocks of 256 bytes (fragment alignment),
// the depth being padded with zeros up to 64.
//   L[row][k] -> out[((row/16)*4 + k/16)*256 + (row%16)*16 + k%16]           (matrix_a, row_major)
//   R[k][col] -> out[((col/16)*4 + k/16)*256 + (col%16)*16 + k%16]           (matrix_b, col_major)
// One thread = one 16-byte row of a packed 16x16 block (stored in a single 16-byte write).
// For L (rows) as for R (columns), the value (idx, k) is src[idx * a + k * b]: L[row][k] = src[row*sr + k*sc]
// (a = sr, b = sc); R[k][col] = src[k*sr + col*sc] (a = sc, b = sr). k >= rk: zero. `idx` (row or column) varies fastest
// between neighbouring threads => coalesced reads.
__global__ void pack16(const int8_t* src, long a, long b, int n, int rk, int8_t* out) {
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= (size_t)n * 4) return;
    src += (size_t)blockIdx.y * n * rk; out += (size_t)blockIdx.y * n * 64; // sigma blockIdx.y of the batch
    const int idx = (int)(t % n), kb = (int)(t / n);
    uint32_t w[4];
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        uint32_t word = 0;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int k = kb * 16 + q * 4 + e;
            const uint32_t v = k < rk ? (uint32_t)(uint8_t)src[(size_t)idx * a + (size_t)k * b] : 0u;
            word |= v << (8 * e);
        }
        w[q] = word;
    }
    *reinterpret_cast<uint4*>(out + ((size_t)(idx >> 4) * 4 + kb) * 256 + (idx & 15) * 16) = make_uint4(w[0], w[1], w[2], w[3]);
}
__global__ void __launch_bounds__(128) noise_tc(const int8_t* x, const int8_t* Lp, const int8_t* Rp, int8_t* out, int n) {
    __shared__ __align__(32) int32_t sc[4][256];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int tiles = n / 16;
    Lp += (size_t)blockIdx.y * n * 64; Rp += (size_t)blockIdx.y * n * 64; out += (size_t)blockIdx.y * n * n; // x (A or B) is common to the batch
    size_t t = (size_t)blockIdx.x * 4 + warp;
    if (t >= (size_t)tiles * tiles) return;
    int ti = (int)(t / tiles), tj = (int)(t % tiles);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;
    wmma::fill_fragment(acc, 0);
    #pragma unroll
    for (int kb = 0; kb < 4; kb++) {
        wmma::load_matrix_sync(af, Lp + ((size_t)ti * 4 + kb) * 256, 16);
        wmma::load_matrix_sync(bf, Rp + ((size_t)tj * 4 + kb) * 256, 16);
        wmma::mma_sync(acc, af, bf, acc);
    }
    wmma::store_matrix_sync(&sc[warp][0], acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 256; e += 32) {
        size_t o = (size_t)(ti * 16 + (e >> 4)) * n + tj * 16 + (e & 15);
        out[o] = (int8_t)((int)x[o] + sc[warp][e]); // |A|<=64 and |E|<=rank<=63: fits in int8 (security_check)
    }
}

// ---------------- tile kernel (v2: CTA 128x128 tiling, smem staging) ----------------
// CTA 128x128 = 8 warps; each warp computes 4x2 sub-tiles 16x16 (WMMA int8) with reuse
// of the fragments. Layout smem "by 16x16 blocks" (256 contiguous B) => 256-bit alignment of the fragments.
// Each 16x16 sub-tile is ONE attempt: folding every fold_k, then blake3.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem) {
    unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// mma.m16n8k32 int8 -> int32. DOCUMENTED layout (PTX ISA): lane = 4*g + t; C: c0,c1 = (row g, columns 2t,2t+1),
// c2,c3 = (row g+8, same columns). This is what allows folding the accumulator DIRECTLY from the registers.
__device__ __forceinline__ void mma_s8(uint32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------- tile kernel (v3: PTX mma m16n8k32, ldmatrix, 4-stage cp.async, folding in registers) ----------------
// CTA 128x128, 8 warps (4 x 2), warp tile 32x64 = 2 x 8 mma m16n8. Each 16x16 block of C is ONE attempt.
// smem: rows of 64 bytes (BK=64) with XOR permutation of the 16 B chunks => ldmatrix without bank conflicts.
#define BM 128
#define BN 128
#define BK 64
#define STAGES 3
#define STAGE_BYTES (BM * BK + BN * BK)
#define SMEM_BYTES (STAGES * STAGE_BYTES)

__device__ __forceinline__ uint32_t swz(int row, int chunk) { return (uint32_t)(row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4)); }

__global__ void __launch_bounds__(256, 2) tile_kernel(const int8_t* __restrict__ ap, const int8_t* __restrict__ btp, int n,
                                                    int fold_k, int tiles, uint32_t* jackpots, int32_t* cout, const uint32_t* hdrs) {
    extern __shared__ __align__(128) uint8_t smem[];
    {   // sigma blockIdx.z of the batch: its noised matrices, its jackpots, its header
        const size_t sb = blockIdx.z;
        ap += sb * n * n; btp += sb * n * n; jackpots += sb * tiles * tiles * 8; hdrs += sb * 12;
    }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1; // 4 x 2 warp grid
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    uint32_t acc[2][8][4];
    uint32_t st[2][4][2][2]; // [m16][column group of 16][half n8][e]: folding words PRIVATE to the lane
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) { acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0; }
        #pragma unroll
        for (int c = 0; c < 4; c++) { st[i][c][0][0] = st[i][c][0][1] = st[i][c][1][0] = st[i][c][1][1] = 0; }
    }

    auto load_stage = [&](int s, int k0) {
        uint8_t* sa = smem + (size_t)s * STAGE_BYTES;
        uint8_t* sb = sa + BM * BK;
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            int p = tid + q * 256, row = p >> 2, ch = p & 3;
            cp_async16(sa + swz(row, ch), &ap[(size_t)(m0 + row) * n + k0 + ch * 16]);
            cp_async16(sb + swz(row, ch), &btp[(size_t)(n0 + row) * n + k0 + ch * 16]);
        }
    };

    const int nsteps = n / BK;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nsteps) load_stage(s, s * BK);
        cp_async_commit();
    }
    for (int it = 0; it < nsteps; it++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads(); // step `it` has arrived AND everybody has finished step it-1 (its buffer is reusable)
        if (it + STAGES - 1 < nsteps) load_stage((it + STAGES - 1) % STAGES, (it + STAGES - 1) * BK);
        cp_async_commit();
        const uint8_t* sa = smem + (size_t)(it % STAGES) * STAGE_BYTES;
        const uint8_t* sb = sa + BM * BK;
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) { // two steps of k=32 per stage
            uint32_t af[2][4], bfr[8][2];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int m = lane >> 3;
                int row = wm * 32 + mi * 16 + (lane & 7) + (m & 1) * 8;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3], sa + swz(row, ks * 2 + (m >> 1)));
            }
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int m = lane >> 3;
                int nrow = wn * 64 + jj * 16 + (lane & 7) + (m >> 1) * 8;
                ldmatrix_x4(bfr[2 * jj][0], bfr[2 * jj][1], bfr[2 * jj + 1][0], bfr[2 * jj + 1][1], sb + swz(nrow, ks * 2 + (m & 1)));
            }
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int nj = 0; nj < 8; nj++) mma_s8(acc[mi][nj], af[mi], bfr[nj]);

            const int kg = it * BK + ks * 32 + 32;
            if (kg % fold_k == 0) {
                const int rot = (kg / fold_k - 1) % 31 + 1;
                // folding word of the element (row r, column c) of a 16x16 attempt: (r % 4) * 16 + c.
                // Here r = g or g+8 (same r % 4 = g % 4), c = 8*(nj&1) + 2t + e  =>  2 values XORed into the same word.
                #pragma unroll
                for (int mi = 0; mi < 2; mi++)
                    #pragma unroll
                    for (int nj = 0; nj < 8; nj++) {
                        st[mi][nj >> 1][nj & 1][0] ^= rotl32(acc[mi][nj][0], rot) ^ rotl32(acc[mi][nj][2], rot);
                        st[mi][nj >> 1][nj & 1][1] ^= rotl32(acc[mi][nj][1], rot) ^ rotl32(acc[mi][nj][3], rot);
                    }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    if (cout) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int nj = 0; nj < 8; nj++)
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    int row = m0 + wm * 32 + mi * 16 + g + (e >> 1) * 8;
                    int col = n0 + wn * 64 + nj * 8 + 2 * t + (e & 1);
                    cout[(size_t)row * n + col] = (int32_t)acc[mi][nj][e];
                }
    }

    // Reduction of the private words: lanes g and g+4 (lane and lane+16) share the same words -> XOR by shuffle.
    uint32_t* sst = reinterpret_cast<uint32_t*>(smem); // 64 attempts x 64 words = 16 KiB (the stages are no longer used)
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int cj = 0; cj < 4; cj++) {
            const int a_id = (wm * 2 + mi) * 8 + (wn * 4 + cj);
            #pragma unroll
            for (int h = 0; h < 2; h++)
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    uint32_t w = st[mi][cj][h][e];
                    w ^= __shfl_xor_sync(0xffffffffu, w, 16);
                    if (lane < 16) sst[a_id * 64 + g * 16 + h * 8 + 2 * t + e] = w;
                }
        }
    __syncthreads();
    if (tid < 64) {
        const int ti = (m0 >> 4) + (tid >> 3), tj = (n0 >> 4) + (tid & 7);
        uint32_t out[8];
        jackpot_hash(hdrs, (uint32_t)ti, (uint32_t)tj, &sst[tid * 64], out);
        for (int e = 0; e < 8; e++) jackpots[((size_t)ti * tiles + tj) * 8 + e] = out[e];
    }
}

// ---------------- API C ----------------
struct Ctx {
    int n, r, cap; // cap = number of sigmas of the batch for which the buffers are allocated
    int8_t *a, *b;                 // A and B^T (resident, common to the whole batch)
    int8_t *ap, *bt, *el, *er, *fl, *fr;
    int8_t *pla, *pra, *plb, *prb; // noise operands packed in 16x16 blocks, depth 64
    uint32_t* jack;
    uint32_t* hdrs;
    int32_t* c;
};

static void free_batch(Ctx* c) {
    cudaFree(c->ap); cudaFree(c->bt); cudaFree(c->el); cudaFree(c->er); cudaFree(c->fl); cudaFree(c->fr);
    cudaFree(c->pla); cudaFree(c->pra); cudaFree(c->plb); cudaFree(c->prb); cudaFree(c->jack); cudaFree(c->hdrs);
    c->ap = c->bt = c->el = c->er = c->fl = c->fr = c->pla = c->pra = c->plb = c->prb = nullptr;
    c->jack = nullptr; c->hdrs = nullptr; c->cap = 0;
}

static bool alloc_batch(Ctx* c, int s) {
    if (s <= c->cap) return true;
    free_batch(c);
    size_t nn = (size_t)c->n * c->n, nr = (size_t)c->n * c->r, S = (size_t)s;
    int tiles = c->n / 16;
    if (cudaMalloc(&c->ap, S * nn) || cudaMalloc(&c->bt, S * nn) ||
        cudaMalloc(&c->el, S * nr) || cudaMalloc(&c->er, S * nr) || cudaMalloc(&c->fl, S * nr) || cudaMalloc(&c->fr, S * nr) ||
        cudaMalloc(&c->pla, S * c->n * 64) || cudaMalloc(&c->pra, S * c->n * 64) || cudaMalloc(&c->plb, S * c->n * 64) ||
        cudaMalloc(&c->prb, S * c->n * 64) ||
        cudaMalloc(&c->jack, S * tiles * tiles * 32) || cudaMalloc(&c->hdrs, S * 48)) { free_batch(c); return false; }
    c->cap = s;
    return true;
}

// Name and compute capability of GPU 0. Returns 0 if OK.
extern "C" int vearl_gpu_device(char* name, int name_len, int* major, int* minor) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return -1;
    snprintf(name, name_len, "%s", p.name);
    *major = p.major;
    *minor = p.minor;
    return 0;
}

static bool gpu_init() {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(tile_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        noise_key_kernel<<<1, 1>>>();
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        attr = true;
    }
    return true;
}

extern "C" void* vearl_gpu_new(const int8_t* a, const int8_t* b, int n, int rank) {
    if (n % 128 != 0 || rank < 1 || rank > 64) return nullptr; // CTA 128x128; noise depth <= 64
    if (!gpu_init()) return nullptr;
    Ctx* c = new Ctx();
    c->n = n; c->r = rank; c->cap = 0;
    c->ap = c->bt = c->el = c->er = c->fl = c->fr = c->pla = c->pra = c->plb = c->prb = nullptr;
    c->jack = nullptr; c->hdrs = nullptr; c->c = nullptr; c->a = nullptr; c->b = nullptr;
    size_t nn = (size_t)n * n;
    if (cudaMalloc(&c->a, nn) || cudaMalloc(&c->b, nn) || cudaMalloc(&c->c, nn * 4) || !alloc_batch(c, 1)) {
        cudaFree(c->a); cudaFree(c->b); cudaFree(c->c); free_batch(c); delete c; return nullptr;
    }
    // B is stored TRANSPOSED (c->b = B^T) once and for all: the columns of B become rows.
    int8_t* bt_host = (int8_t*)malloc(nn);
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++) bt_host[(size_t)j * n + k] = b[(size_t)k * n + j];
    cudaMemcpy(c->a, a, nn, cudaMemcpyHostToDevice);
    cudaMemcpy(c->b, bt_host, nn, cudaMemcpyHostToDevice);
    free(bt_host);
    return c;
}

extern "C" void vearl_gpu_free(void* p) {
    Ctx* c = (Ctx*)p;
    if (!c) return;
    cudaFree(c->a); cudaFree(c->b); cudaFree(c->c);
    free_batch(c);
    delete c;
}

// Batch of `s` sigmas in a single chain of launches (the inputs are concatenated: sigma i has its slices
// hdr48[48 i..], el[n r i..], ...). jackpots_out: s * tiles * tiles * 32 bytes; c_out (nullable, s == 1 only): n*n int32.
// ms_out[0]=noise, [1]=tiles.
static int run_impl(Ctx* c, int s, const uint8_t* hdr48, const int8_t* el, const int8_t* er, const int8_t* fl,
                    const int8_t* fr, int fold_k, uint8_t* jackpots_out, int32_t* c_out, float* ms_out, bool gen_noise) {
    int n = c->n, r = c->r, tiles = n / 16;
    if (fold_k % 32 != 0 || n % fold_k != 0) return -2; // the kernel folds after steps of k=32
    if (s < 1 || s > 65535 || (c_out && s != 1)) return -3;
    if (!alloc_batch(c, s)) return -4;
    size_t nr = (size_t)n * r * s;
    if (!gen_noise) {
        CK(cudaMemcpy(c->el, el, nr, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->er, er, nr, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->fl, fl, nr, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->fr, fr, nr, cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(c->hdrs, hdr48, (size_t)48 * s, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventRecord(e0);
    if (gen_noise) {
        size_t nr1 = (size_t)n * r;
        dim3 gg((unsigned)((4 * (nr1 / 64) + 127) / 128), s);
        noise_gen<<<gg, 128>>>(c->hdrs, c->el, c->er, c->fl, c->fr, nr1);
    }
    unsigned pkb = (unsigned)(((size_t)n * 4 + 127) / 128);
    dim3 pg(pkb, s);
    pack16<<<pg, 128>>>(c->el, r, 1, n, r, c->pla);  // A : L[i][t]  = EL[i][t]  = el[i*r + t]
    pack16<<<pg, 128>>>(c->er, 1, n, n, r, c->pra);  //     R[t][k]  = ER[t][k]  = er[t*n + k]
    pack16<<<pg, 128>>>(c->fr, 1, n, n, r, c->plb);  // B^T : L[j][t] = FR[t][j] = fr[t*n + j]
    pack16<<<pg, 128>>>(c->fl, r, 1, n, r, c->prb);  //       R[t][k] = FL[k][t] = fl[k*r + t]
    dim3 ng((unsigned)(((size_t)tiles * tiles + 3) / 4), s);
    noise_tc<<<ng, 128>>>(c->a, c->pla, c->pra, c->ap, n);
    noise_tc<<<ng, 128>>>(c->b, c->plb, c->prb, c->bt, n);
    cudaEventRecord(e1);
    dim3 grid(n / BN, n / BM, s);
    tile_kernel<<<grid, 256, SMEM_BYTES>>>(c->ap, c->bt, n, fold_k, tiles, c->jack, c_out ? c->c : nullptr, c->hdrs);
    cudaEventRecord(e2);
    CK(cudaGetLastError());
    CK(cudaEventSynchronize(e2));
    if (ms_out) { cudaEventElapsedTime(&ms_out[0], e0, e1); cudaEventElapsedTime(&ms_out[1], e1, e2); }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    CK(cudaMemcpy(jackpots_out, c->jack, (size_t)s * tiles * tiles * 32, cudaMemcpyDeviceToHost));
    if (c_out) CK(cudaMemcpy(c_out, c->c, (size_t)n * n * 4, cudaMemcpyDeviceToHost));
    return 0;
}

// hdr48 = "vearl/jackpot/v2" || sigma.
extern "C" int vearl_gpu_run(void* p, const uint8_t* hdr48, const int8_t* el, const int8_t* er, const int8_t* fl,
                             const int8_t* fr, int fold_k, uint8_t* jackpots_out, int32_t* c_out, float* ms_out) {
    return run_impl((Ctx*)p, 1, hdr48, el, er, fl, fr, fold_k, jackpots_out, c_out, ms_out, false);
}

extern "C" int vearl_gpu_run_batch(void* p, int s, const uint8_t* hdr48, const int8_t* el, const int8_t* er, const int8_t* fl,
                                   const int8_t* fr, int fold_k, uint8_t* jackpots_out, float* ms_out) {
    return run_impl((Ctx*)p, s, hdr48, el, er, fl, fr, fold_k, jackpots_out, nullptr, ms_out, false);
}

// Same thing, but the noise is generated on the GPU from the headers alone (hdr48 = domain || sigma, per sigma).
extern "C" int vearl_gpu_run_batch_gen(void* p, int s, const uint8_t* hdr48, int fold_k, uint8_t* jackpots_out, float* ms_out) {
    return run_impl((Ctx*)p, s, hdr48, nullptr, nullptr, nullptr, nullptr, fold_k, jackpots_out, nullptr, ms_out, true);
}

// ---------------- EXACT int8 GEMM (C = A*B, int32): the USEFUL computation of a job, on tensor cores ----------------
// Reuses the tile kernel without noise (sigma = 1, no folding) with output of C; the transposition of B is done on the GPU.
__global__ void transpose_i8(const int8_t* __restrict__ src, int8_t* __restrict__ dst, int n) {
    __shared__ int8_t t[32][33];
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) t[threadIdx.y + j][threadIdx.x] = src[(size_t)(y + j) * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x; y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) dst[(size_t)(y + j) * n + x] = t[threadIdx.x][threadIdx.y + j];
}

// n multiple of 128; a, b: n*n int8 (row-major); c_out: n*n int32. Returns 0 if OK.
extern "C" int vearl_gpu_exact_gemm(const int8_t* a, const int8_t* b, int n, int32_t* c_out) {
    if (n % 128 != 0 || n < 128 || !gpu_init()) return -1;
    size_t nn = (size_t)n * n;
    int tiles = n / 16;
    int8_t *da = nullptr, *db = nullptr, *dbt = nullptr;
    int32_t* dc = nullptr;
    uint32_t *jack = nullptr, *hdrs = nullptr;
    int rc = 0;
    if (cudaMalloc(&da, nn) || cudaMalloc(&db, nn) || cudaMalloc(&dbt, nn) || cudaMalloc(&dc, nn * 4) ||
        cudaMalloc(&jack, (size_t)tiles * tiles * 32) || cudaMalloc(&hdrs, 48)) rc = -2;
    if (!rc) {
        cudaMemset(hdrs, 0, 48);
        cudaMemcpy(da, a, nn, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, nn, cudaMemcpyHostToDevice);
        dim3 tg(n / 32, n / 32), tb(32, 8);
        transpose_i8<<<tg, tb>>>(db, dbt, n);
        dim3 grid(n / BN, n / BM, 1);
        tile_kernel<<<grid, 256, SMEM_BYTES>>>(da, dbt, n, 1 << 30, tiles, jack, dc, hdrs);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) rc = -3;
        else cudaMemcpy(c_out, dc, nn * 4, cudaMemcpyDeviceToHost);
    }
    cudaFree(da); cudaFree(db); cudaFree(dbt); cudaFree(dc); cudaFree(jack); cudaFree(hdrs);
    return rc;
}
