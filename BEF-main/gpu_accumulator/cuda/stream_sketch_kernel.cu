
#include <torch/extension.h>
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

__global__ void dot_mod_p31_kernel(
    const long long* __restrict__ vals,
    const long long* __restrict__ rpow,
    unsigned long long* __restrict__ out_partial,
    int N){
  extern __shared__ unsigned long long sh[];
  int tid = threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + tid;
  unsigned long long acc = 0ULL;
  while (idx < N){
    unsigned long long a = static_cast<unsigned long long>(vals[idx]) % MODULUS;
    unsigned long long b = static_cast<unsigned long long>(rpow[idx]) % MODULUS;
    acc = add_mod(acc, mul_mod(a, b));
    idx += stride;
  }
  sh[tid] = acc; __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1){
    if (tid < off){ sh[tid] = add_mod(sh[tid], sh[tid + off]); }
    __syncthreads();
  }
  if (tid == 0){ out_partial[blockIdx.x] = sh[0]; }
}

extern "C" torch::Tensor dot_mod_p31_launcher(
    torch::Tensor vals,
    torch::Tensor rpow,
    int block_size,
    int max_blocks){
  TORCH_CHECK(vals.is_cuda() && rpow.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(vals.dtype() == torch::kInt64 && rpow.dtype() == torch::kInt64, "inputs must be int64");
  const int64_t N64 = vals.size(0); TORCH_CHECK(rpow.size(0) == N64, "size mismatch");
  int N = static_cast<int>(N64);
  int blocks = (N + block_size - 1) / block_size; if (blocks > max_blocks) blocks = max_blocks;
  auto opts = torch::TensorOptions().device(vals.device()).dtype(torch::kInt64);
  torch::Tensor out = torch::empty({blocks}, opts);
  size_t shmem = static_cast<size_t>(block_size) * sizeof(unsigned long long);
  dot_mod_p31_kernel<<<blocks, block_size, shmem>>>(
      reinterpret_cast<const long long*>(vals.data_ptr<int64_t>()),
      reinterpret_cast<const long long*>(rpow.data_ptr<int64_t>()),
      reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()),
      N);
  return out;
}

// Fill r^i table using block tiling: out[j*BS + k] = anchors[j] * base[k] mod MODULUS
__global__ void fill_rpow_blocks_kernel(
    const unsigned long long* __restrict__ base,   // [BS]
    const long long* __restrict__ anchors,         // [blocks]
    long long* __restrict__ out,                   // [L]
    int BS,
    int blocks,
    int L){
  int j = blockIdx.x;         // block index
  int k = threadIdx.x;        // within-tile index
  if (j >= blocks || k >= BS) return;
  int idx = j * BS + k;
  if (idx >= L) return;
  unsigned long long a = static_cast<unsigned long long>(anchors[j]) % MODULUS;
  unsigned long long b = base[k] % MODULUS;
  unsigned long long val = mul_mod(a, b);
  out[idx] = static_cast<long long>(val);
}

extern "C" void fill_rpow_blocks_launcher(
    torch::Tensor base,      // int64 CUDA [BS]
    torch::Tensor anchors,   // int64 CUDA [blocks]
    torch::Tensor out,       // int64 CUDA [L]
    int BS){
  TORCH_CHECK(base.is_cuda() && anchors.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(base.dtype()==torch::kInt64 && anchors.dtype()==torch::kInt64 && out.dtype()==torch::kInt64, "tensors must be int64");
  int blocks = static_cast<int>(anchors.size(0));
  int L = static_cast<int>(out.size(0));
  dim3 grid(blocks), blk(BS);
  fill_rpow_blocks_kernel<<<grid, blk>>>(
      reinterpret_cast<const unsigned long long*>(base.data_ptr<int64_t>()),
      reinterpret_cast<const long long*>(anchors.data_ptr<int64_t>()),
      reinterpret_cast<long long*>(out.data_ptr<int64_t>()),
      BS, blocks, L);
}

// =========================
// Fused multi-challenge path
// =========================

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

#ifndef FUSED_K_MAX
#define FUSED_K_MAX 8
#endif

__global__ void fused_sketch_kernel(
    const long long* __restrict__ vals,
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

  for (int i = gid; i < N; i += stride){
    unsigned long long v = static_cast<unsigned long long>(vals[i]);
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
  for (int j = 0; j < k; ++j){
    unsigned long long local = 0ULL;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x){
      local = add_mod(local, block_partials[b * k + j]);
    }
    sh[threadIdx.x] = local;
    __syncthreads();
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

extern "C" void fused_sketch_launcher(
    torch::Tensor vals,
    torch::Tensor challenges,
    torch::Tensor out_S,
    int block_size,
    int max_blocks){
  TORCH_CHECK(vals.is_cuda(), "vals must be CUDA tensor");
  TORCH_CHECK(vals.dtype() == torch::kInt64, "vals must be int64");
  TORCH_CHECK(challenges.dtype() == torch::kInt64, "challenges must be int64");
  TORCH_CHECK(out_S.is_cuda() && out_S.dtype() == torch::kInt64, "output must be CUDA int64");

  int64_t N64 = vals.size(0);
  int N = static_cast<int>(N64);
  int k = static_cast<int>(challenges.size(0));
  TORCH_CHECK(k > 0 && k <= FUSED_K_MAX, "num challenges out of range");

  int blocks = (N + block_size - 1) / block_size;
  if (blocks < 1) blocks = 1;
  if (blocks > max_blocks) blocks = max_blocks;

  torch::Tensor chall_dev = challenges.is_cuda() ? challenges : challenges.to(vals.device());
  auto opts = torch::TensorOptions().device(vals.device()).dtype(torch::kInt64);
  torch::Tensor partials = torch::empty({blocks * k}, opts);

  out_S.zero_();

  fused_sketch_kernel<<<blocks, block_size>>>(
      reinterpret_cast<const long long*>(vals.data_ptr<int64_t>()),
      N,
      reinterpret_cast<const unsigned long long*>(chall_dev.data_ptr<int64_t>()),
      k,
      reinterpret_cast<unsigned long long*>(partials.data_ptr<int64_t>())
  );

  int reduce_threads = 256;
  size_t shmem = static_cast<size_t>(reduce_threads) * sizeof(unsigned long long);
  reduce_blocks_kernel<<<1, reduce_threads, shmem>>>(
      reinterpret_cast<const unsigned long long*>(partials.data_ptr<int64_t>()),
      blocks,
      k,
      reinterpret_cast<unsigned long long*>(out_S.data_ptr<int64_t>())
  );
}

// (duplicate block removed)
