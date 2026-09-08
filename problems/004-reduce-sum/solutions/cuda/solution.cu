// 004 归约求和 —— CUDA 实现（两级归约 + warp shuffle）
// 阶段1：每个 block 归约一段连续数据 -> partial[blockIdx]
// 阶段2：单个 block 归并所有 partial -> out
// 对应 Mark Harris《Optimizing Parallel Reduction in CUDA》的 tree/shuffle 版本。
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

constexpr int kBlock = 256;
constexpr int kItemsPerThread = 8;  // 每线程先串行累加 8 个元素再进树形归约
constexpr int kWarps = kBlock / 32;

struct SumOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

// block 内树形归约：warp 内 shuffle（寄存器交换，零共享内存流量），warp 间共享内存。
// 返回值仅 threadIdx.x == 0 有效。
// 注意：同一 block 内连续调用两次本函数时，函数入口的 __syncthreads()
// 保证第二次写入 warp_sums 前上一次读取已结束，避免脏读。
template <typename Op>
__device__ __forceinline__ float block_reduce(float v, Op op, float identity) {
  __shared__ float warp_sums[kWarps];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;

  for (int s = 16; s > 0; s >>= 1) v = op(v, __shfl_down_sync(0xffffffffu, v, s));
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();

  float out = identity;
  if (warp == 0) {
    if (lane < kWarps) out = warp_sums[lane];
    for (int s = 16; s > 0; s >>= 1) out = op(out, __shfl_down_sync(0xffffffffu, out, s));
  }
  return out;
}

// 阶段1：block 内先跨步读 kItemsPerThread 个元素（相邻线程访问相邻地址 => 合并访存）
__global__ void reduce_partial_kernel(const float* __restrict__ x,
                                      float* __restrict__ partial, long n) {
  const long base = (long)blockIdx.x * kBlock * kItemsPerThread;
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kItemsPerThread; ++k) {
    const long i = base + (long)k * kBlock + threadIdx.x;
    if (i < n) acc += x[i];
  }
  acc = block_reduce(acc, SumOp(), 0.0f);
  if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}

// 阶段2：单 block 归并全部部分和
__global__ void reduce_final_kernel(const float* __restrict__ partial,
                                    float* __restrict__ out, long m) {
  float acc = 0.0f;
  for (long i = threadIdx.x; i < m; i += kBlock) acc += partial[i];
  acc = block_reduce(acc, SumOp(), 0.0f);
  if (threadIdx.x == 0) *out = acc;
}

torch::Tensor reduce_sum(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "only fp32 is supported");
  auto xc = x.contiguous().view({-1});
  const long n = xc.numel();
  auto out = torch::zeros({}, x.options());
  if (n == 0) return out;

  const long seg = (long)kBlock * kItemsPerThread;
  const long blocks = (n + seg - 1) / seg;
  auto partial = torch::empty({blocks}, x.options());

  reduce_partial_kernel<<<(unsigned int)blocks, kBlock>>>(
      xc.data_ptr<float>(), partial.data_ptr<float>(), n);
  reduce_final_kernel<<<1, kBlock>>>(partial.data_ptr<float>(),
                                     out.data_ptr<float>(), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum", &reduce_sum, "reduce sum (CUDA, two-pass)", py::arg("x"));
}
