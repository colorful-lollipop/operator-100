// 006 Softmax —— CUDA 实现（一个 block 负责一行，safe softmax）
// fp16 输入在寄存器里转 fp32 计算再转回——LLM 推理 kernel 的标准姿势。
// 注意 block_reduce 模板函数定义在共享头文件之外，这里直接内联展开。
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Half.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---- 可复用的 block 内 max / sum 归约（warp shuffle + 共享内存）----
constexpr int kMaxBlock = 1024;
constexpr int kMaxWarps = kMaxBlock / 32;

// block 内归约三步：warp 内 shuffle -> warp 间共享内存 -> 广播回全 block。
// 关键：最终结果只由 warp0 算出，必须写回共享内存让所有线程拿到同一个值——
// 忘记广播是行归约 kernel 最经典的错误（004 里只有 thread0 用结果所以侥幸没炸）。
// 入口 __syncthreads()：同一 kernel 连续调用两次归约时防止共享内存复用竞态。
template <int BLOCK>
__device__ __forceinline__ float block_max(float v) {
  constexpr int kWarps = BLOCK / 32;
  __shared__ float warp_sums[kWarps];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __syncthreads();
  for (int s = 16; s > 0; s >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, s));
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float out = (lane < kWarps) ? warp_sums[lane] : -INFINITY;
    for (int s = 16; s > 0; s >>= 1) out = fmaxf(out, __shfl_down_sync(0xffffffffu, out, s));
    if (lane == 0) warp_sums[0] = out;  // 广播位
  }
  __syncthreads();
  return warp_sums[0];
}

template <int BLOCK>
__device__ __forceinline__ float block_sum(float v) {
  constexpr int kWarps = BLOCK / 32;
  __shared__ float warp_sums[kWarps];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __syncthreads();
  for (int s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float out = (lane < kWarps) ? warp_sums[lane] : 0.0f;
    for (int s = 16; s > 0; s >>= 1) out += __shfl_down_sync(0xffffffffu, out, s);
    if (lane == 0) warp_sums[0] = out;  // 广播位
  }
  __syncthreads();
  return warp_sums[0];
}

// ---- fp32 行 softmax：block 按行跨步扫描 ----
template <int BLOCK>
__global__ void softmax_fp32_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, long cols) {
  const float* xr = x + (long)blockIdx.x * cols;
  float* yr = y + (long)blockIdx.x * cols;

  float m = -INFINITY;
  for (long i = threadIdx.x; i < cols; i += BLOCK) m = fmaxf(m, xr[i]);
  m = block_max<BLOCK>(m);

  float s = 0.0f;
  for (long i = threadIdx.x; i < cols; i += BLOCK) {
    const float e = expf(xr[i] - m);
    yr[i] = e;  // 先暂存 exp 值，归一化后覆写
    s += e;
  }
  s = block_sum<BLOCK>(s);
  const float inv = 1.0f / s;
  for (long i = threadIdx.x; i < cols; i += BLOCK) yr[i] *= inv;
}

// ---- fp16 行 softmax：fp32 累加 ----
template <int BLOCK>
__global__ void softmax_fp16_kernel(const __half* __restrict__ x, __half* __restrict__ y,
                                    long cols) {
  const __half* xr = x + (long)blockIdx.x * cols;
  __half* yr = y + (long)blockIdx.x * cols;

  float m = -INFINITY;
  for (long i = threadIdx.x; i < cols; i += BLOCK) m = fmaxf(m, __half2float(xr[i]));
  m = block_max<BLOCK>(m);

  float s = 0.0f;
  for (long i = threadIdx.x; i < cols; i += BLOCK) {
    const float e = expf(__half2float(xr[i]) - m);
    yr[i] = __float2half(e);  // 暂存
    s += e;
  }
  s = block_sum<BLOCK>(s);
  const float inv = 1.0f / s;
  for (long i = threadIdx.x; i < cols; i += BLOCK) yr[i] = __float2half(__half2float(yr[i]) * inv);
}

torch::Tensor softmax(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, cols]");
  const bool is_fp32 = x.scalar_type() == at::kFloat;
  const bool is_fp16 = x.scalar_type() == at::kHalf;
  TORCH_CHECK(is_fp32 || is_fp16, "fp32 / fp16 supported");
  auto xc = x.contiguous();
  const long rows = xc.size(0), cols = xc.size(1);
  auto y = torch::empty_like(xc);
  if (rows == 0 || cols == 0) return y;
  TORCH_CHECK(cols <= 16384, "cols too large; use online softmax (problem 031)");

  // block 大小：不小于 64、不超过 1024 的 2 的幂
  int block = 64;
  while (block < cols && block < 1024) block <<= 1;

#define DISPATCH_SOFTMAX(B)                                                     \
  do {                                                                          \
    if (is_fp32)                                                                \
      softmax_fp32_kernel<B><<<(unsigned int)rows, B>>>(                        \
          xc.data_ptr<float>(), y.data_ptr<float>(), cols);                     \
    else                                                                        \
      softmax_fp16_kernel<B><<<(unsigned int)rows, B>>>(                        \
          reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),             \
          reinterpret_cast<__half*>(y.data_ptr<at::Half>()), cols);             \
  } while (0)

  switch (block) {
    case 64: DISPATCH_SOFTMAX(64); break;
    case 128: DISPATCH_SOFTMAX(128); break;
    case 256: DISPATCH_SOFTMAX(256); break;
    case 512: DISPATCH_SOFTMAX(512); break;
    default: DISPATCH_SOFTMAX(1024); break;
  }
#undef DISPATCH_SOFTMAX
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &softmax, "row softmax (CUDA, safe softmax)", py::arg("x"));
}
