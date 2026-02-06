#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Define MODULUS if not defined
#ifndef MODULUS
#define MODULUS 2305843009213693951ULL  // 2^61 - 1
#endif

#define MOD_BITS 61
#define FUSED_K_MAX 8

// --- Device Helpers ---

__device__ __forceinline__ unsigned long long add_mod(unsigned long long a, unsigned long long b){
  unsigned long long s = a + b;
  if (s >= MODULUS) s -= MODULUS;
  return s;
}

__device__ __forceinline__ unsigned long long fold_mod128(unsigned __int128 x){
  unsigned long long lo = static_cast<unsigned long long>(x & MODULUS);
  unsigned long long hi = static_cast<unsigned long long>(x >> MOD_BITS);
  unsigned long long s = lo + hi;
  s = (s & MODULUS) + (s >> MOD_BITS);
  if (s >= MODULUS) s -= MODULUS;
  return s;
}

__device__ __forceinline__ unsigned long long mul_mod(unsigned long long a, unsigned long long b){
  unsigned __int128 prod = static_cast<unsigned __int128>(a & MODULUS) * static_cast<unsigned __int128>(b & MODULUS);
  return fold_mod128(prod);
}

__device__ __forceinline__ unsigned long long pow_mod_p61(unsigned long long base, unsigned long long exp){
  unsigned long long res = 1ULL;
  while (exp){
    if (exp & 1ULL){
      unsigned __int128 prod = static_cast<unsigned __int128>(res & MODULUS) * static_cast<unsigned __int128>(base & MODULUS);
      res = fold_mod128(prod);
    }
    unsigned __int128 sq = static_cast<unsigned __int128>(base & MODULUS) * static_cast<unsigned __int128>(base & MODULUS);
    base = fold_mod128(sq);
    exp >>= 1ULL;
  }
  return res;
}

__device__ __forceinline__ unsigned long long warp_reduce_sum_mod(unsigned long long x){
  unsigned mask = 0xffffffffu;
  x = add_mod(x, __shfl_down_sync(mask, x, 16));
  x = add_mod(x, __shfl_down_sync(mask, x, 8));
  x = add_mod(x, __shfl_down_sync(mask, x, 4));
  x = add_mod(x, __shfl_down_sync(mask, x, 2));
  x = add_mod(x, __shfl_down_sync(mask, x, 1));
  return x;
}

// --- Kernels ---

__global__ void fused_sketch_kernel(
    const unsigned long long* __restrict__ vals,
    int N,
    const unsigned long long* __restrict__ r_vec,
    int k,
    unsigned long long* __restrict__ block_partials)
{
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warpId = tid >> 5;
  int warpsPerBlock = blockDim.x >> 5;
  int gid = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  if (k <= 0 || k > FUSED_K_MAX) return;

  unsigned long long acc[FUSED_K_MAX];
  unsigned long long rpow[FUSED_K_MAX];
  unsigned long long rstep[FUSED_K_MAX];

  #pragma unroll
  for (int j = 0; j < FUSED_K_MAX; ++j){
    if (j < k){ acc[j] = 0ULL; }
  }

  // Initialize powers for this thread
  #pragma unroll
  for (int j = 0; j < FUSED_K_MAX; ++j){
    if (j < k){
      unsigned long long rj = r_vec[j] % MODULUS;
      if (rj == 0ULL) rj = 1ULL;
      unsigned long long gid_u = static_cast<unsigned long long>(gid < 0 ? 0 : gid);
      unsigned long long stride_u = static_cast<unsigned long long>(stride <= 0 ? 1 : stride);
      rpow[j] = pow_mod_p61(rj, gid_u);
      rstep[j] = pow_mod_p61(rj, stride_u);
    }
  }

  // Grid-stride loop
  for (int i = gid; i < N; i += stride){
    unsigned long long v = vals[i];
    if (v >= MODULUS) v %= MODULUS;
    #pragma unroll
    for (int j = 0; j < FUSED_K_MAX; ++j){
      if (j < k){
        unsigned __int128 prod = static_cast<unsigned __int128>(v & MODULUS) * static_cast<unsigned __int128>(rpow[j] & MODULUS);
        acc[j] = add_mod(acc[j], fold_mod128(prod));
        rpow[j] = mul_mod(rpow[j], rstep[j]);
      }
    }
  }

  // Warp reduce
  #pragma unroll
  for (int j = 0; j < FUSED_K_MAX; ++j){
    if (j < k){ acc[j] = warp_reduce_sum_mod(acc[j]); }
  }

  __shared__ unsigned long long smem[FUSED_K_MAX * 32];
  if (lane == 0){
    #pragma unroll
    for (int j = 0; j < FUSED_K_MAX; ++j){
      if (j < k){ smem[j * 32 + warpId] = acc[j]; }
    }
  }
  __syncthreads();

  // Block reduce (first warp)
  if (warpId == 0){
    #pragma unroll
    for (int j = 0; j < FUSED_K_MAX; ++j){
      if (j < k){
        unsigned long long x = (lane < warpsPerBlock) ? smem[j * 32 + lane] : 0ULL;
        x = warp_reduce_sum_mod(x);
        if (lane == 0){ block_partials[blockIdx.x * k + j] = x; }
      }
    }
  }
}

__global__ void reduce_blocks_kernel(
    const unsigned long long* __restrict__ block_partials,
    int num_blocks,
    int k,
    unsigned long long* __restrict__ out_S)
{
  extern __shared__ unsigned long long sh[];
  // Each thread handles one challenge j? No, k is small (e.g. 2, 4, 8)
  // Let's parallelize across challenges and blocks?
  // Current kernel logic from original file:
  // "for j < k" loop, but inside threads work on blocks.
  // "for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)"
  
  // Outer loop is j (challenges)
  for (int j = 0; j < k; ++j){
    unsigned long long local = 0ULL;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x){
      local = add_mod(local, block_partials[b * k + j]);
    }
    sh[threadIdx.x] = local;
    __syncthreads();
    
    // Block reduce
    for (int off = blockDim.x >> 1; off > 0; off >>= 1){
      if (threadIdx.x < off){
        sh[threadIdx.x] = add_mod(sh[threadIdx.x], sh[threadIdx.x + off]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0){
      out_S[j] = sh[0];
    }
    __syncthreads();
  }
}

// --- C-ABI Interface ---

extern "C" int gpu_compute_sketches(
    const uint64_t* input_data,
    size_t input_len,
    const uint64_t* challenges,
    size_t num_challenges,
    uint64_t* output_sketches
) {
    if (num_challenges > FUSED_K_MAX) return -1; // Error: too many challenges

    unsigned long long* d_vals = nullptr;
    unsigned long long* d_chall = nullptr;
    unsigned long long* d_out = nullptr;
    unsigned long long* d_partials = nullptr;

    size_t vals_bytes = input_len * sizeof(uint64_t);
    size_t chall_bytes = num_challenges * sizeof(uint64_t);
    size_t out_bytes = num_challenges * sizeof(uint64_t);

    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc(&d_vals, vals_bytes);
    if (err != cudaSuccess) return -2;
    err = cudaMalloc(&d_chall, chall_bytes);
    if (err != cudaSuccess) { cudaFree(d_vals); return -2; }
    err = cudaMalloc(&d_out, out_bytes);
    if (err != cudaSuccess) { cudaFree(d_vals); cudaFree(d_chall); return -2; }

    // Copy inputs
    cudaMemcpy(d_vals, input_data, vals_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_chall, challenges, chall_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_bytes);

    // Launch configuration
    int block_size = 512;
    int max_blocks = 1024; // Reasonable limit
    int blocks = (input_len + block_size - 1) / block_size;
    if (blocks == 0) blocks = 1;
    if (blocks > max_blocks) blocks = max_blocks;

    size_t partials_bytes = blocks * num_challenges * sizeof(uint64_t);
    err = cudaMalloc(&d_partials, partials_bytes);
    if (err != cudaSuccess) { 
        cudaFree(d_vals); cudaFree(d_chall); cudaFree(d_out); return -2; 
    }

    fused_sketch_kernel<<<blocks, block_size>>>(
        d_vals,
        (int)input_len,
        d_chall,
        (int)num_challenges,
        d_partials
    );

    int reduce_threads = 256;
    size_t shmem = reduce_threads * sizeof(unsigned long long);
    reduce_blocks_kernel<<<1, reduce_threads, shmem>>>(
        d_partials,
        blocks,
        (int)num_challenges,
        d_out
    );

    // Copy result back
    cudaMemcpy(output_sketches, d_out, out_bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_vals);
    cudaFree(d_chall);
    cudaFree(d_out);
    cudaFree(d_partials);

    return 0; // Success
}
