// 007 RMSNorm —— CUDA 实现（一个 block 一行；fp16 输入 fp32 累加）
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// block 内求和归约三步：warp 内 shuffle -> warp 间共享内存 -> 广播回全 block。
// 最终结果只由 warp0 算出，必须写回共享内存让所有线程拿到同一个值，
// 否则只有 warp0 的线程能拿到正确缩放系数（经典错误，004 里因只有 thread0
// 用结果而侥幸正确，本题需要全 block 广播）。
template <int BLOCK>
__device__ __forceinline__ float block_sum(float v) {
  constexpr int kWarps = BLOCK / 32;
  __shared__ float warp_sums[kWarps];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __syncthreads();  // 入口屏障：连续多次归约时防止共享内存复用竞态
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

// 两遍扫描：先平方和，再统一写回。sumsq/H + eps 后取 rsqrt。
template <int BLOCK>
__global__ void rms_norm_fp32_kernel(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ y, long cols, float eps) {
  const float* xr = x + (long)blockIdx.x * cols;
  float* yr = y + (long)blockIdx.x * cols;
  float acc = 0.0f;
  for (long i = threadIdx.x; i < cols; i += BLOCK) acc += xr[i] * xr[i];
  acc = block_sum<BLOCK>(acc);
  const float inv = rsqrtf(acc / cols + eps);  // block_sum 已广播：全 block 拿到同一个值
  for (long i = threadIdx.x; i < cols; i += BLOCK) yr[i] = xr[i] * inv * w[i];
}

template <int BLOCK>
__global__ void rms_norm_fp16_kernel(const __half* __restrict__ x,
                                     const float* __restrict__ w,
                                     __half* __restrict__ y, long cols, float eps) {
  const __half* xr = x + (long)blockIdx.x * cols;
  __half* yr = y + (long)blockIdx.x * cols;
  // 第一遍：fp32 平方和（fp16 直接累加必溢出）
  float acc = 0.0f;
  for (long i = threadIdx.x; i < cols; i += BLOCK) {
    const float v = __half2float(xr[i]);
    acc += v * v;
  }
  acc = block_sum<BLOCK>(acc);
  const float inv = rsqrtf(acc / cols + eps);
  for (long i = threadIdx.x; i < cols; i += BLOCK) {
    yr[i] = __float2half(__half2float(xr[i]) * inv * w[i]);
  }
}

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x/w must be CUDA tensors");
  TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be fp32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, H]");
  const bool is_fp32 = x.scalar_type() == at::kFloat;
  const bool is_fp16 = x.scalar_type() == at::kHalf;
  TORCH_CHECK(is_fp32 || is_fp16, "fp32 / fp16 supported");
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  const long rows = xc.size(0), cols = xc.size(1);
  auto y = torch::empty_like(xc);
  if (rows == 0 || cols == 0) return y;

  int block = 64;
  while (block < cols && block < 1024) block <<= 1;

#define DISPATCH_RMS(B)                                                        \
  do {                                                                         \
    if (is_fp32)                                                               \
      rms_norm_fp32_kernel<B><<<(unsigned int)rows, B>>>(                      \
          xc.data_ptr<float>(), wc.data_ptr<float>(), y.data_ptr<float>(),     \
          cols, (float)eps);                                                   \
    else                                                                       \
      rms_norm_fp16_kernel<B><<<(unsigned int)rows, B>>>(                      \
          reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),            \
          wc.data_ptr<float>(),                                                \
          reinterpret_cast<__half*>(y.data_ptr<at::Half>()), cols, (float)eps);\
  } while (0)

  switch (block) {
    case 64: DISPATCH_RMS(64); break;
    case 128: DISPATCH_RMS(128); break;
    case 256: DISPATCH_RMS(256); break;
    case 512: DISPATCH_RMS(512); break;
    default: DISPATCH_RMS(1024); break;
  }
#undef DISPATCH_RMS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm, "RMSNorm (CUDA)", py::arg("x"), py::arg("w"),
        py::arg("eps") = 1e-5);
}
