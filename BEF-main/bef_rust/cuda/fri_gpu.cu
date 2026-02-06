#include <cuda_runtime.h>
#include <cstdint>

#ifndef MODULUS
#define MODULUS 2305843009213693951ULL  // 2^61 - 1
#endif
#define MOD_BITS 61

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

__global__ void fold_fri_kernel(
    const unsigned long long* __restrict__ values,
    int n, // Input length (must be even)
    unsigned long long alpha,
    unsigned long long* __restrict__ next_values
) {
    int next_n = n / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < next_n; i += stride) {
        unsigned long long a = values[2 * i];
        unsigned long long b = values[2 * i + 1];
        // next = a + alpha * b
        unsigned long long term = mul_mod(alpha, b);
        next_values[i] = add_mod(a, term);
    }
}

extern "C" int gpu_fold_fri(
    const uint64_t* input_values,
    size_t input_len,
    uint64_t alpha,
    uint64_t* output_values
) {
    if (input_len % 2 != 0) return -1;
    size_t next_len = input_len / 2;
    
    unsigned long long* d_in = nullptr;
    unsigned long long* d_out = nullptr;
    
    size_t in_bytes = input_len * sizeof(uint64_t);
    size_t out_bytes = next_len * sizeof(uint64_t);
    
    cudaError_t err;
    err = cudaMalloc(&d_in, in_bytes);
    if (err != cudaSuccess) return -2;
    
    err = cudaMalloc(&d_out, out_bytes);
    if (err != cudaSuccess) { cudaFree(d_in); return -2; }
    
    cudaMemcpy(d_in, input_values, in_bytes, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int blocks = (next_len + block_size - 1) / block_size;
    if (blocks > 1024) blocks = 1024; // Grid stride loop handles rest
    
    fold_fri_kernel<<<blocks, block_size>>>(d_in, (int)input_len, alpha, d_out);
    
    cudaMemcpy(output_values, d_out, out_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
