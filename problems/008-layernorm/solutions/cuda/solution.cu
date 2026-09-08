// 008 LayerNorm —— CUDA 实现（一个 block 一行，两遍扫描求均值/方差）
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <int BLOCK>
__device__ __forceinline__ float block_sum(float v) {
  constexpr int kWarps = BLOCK / 32;
  __shared__ float warp_sums[kWarps];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  float out = 0.0f;
  if (warp == 0) {
    if (lane < kWarps) out = warp_sums[lane];
    for (int s = 16; s > 0; s >>= 1) out += __shfl_down_sync(0xffffffffu, out, s);
  }
  return out;
}

// fp32：两遍扫描（μ 与 σ²），第三遍写回。数值稳定优于一遍式 E[x²]-μ²。
template <int BLOCK>
__global__ void layer_norm_fp32_kernel(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       const float* __restrict__ b,
                                       float* __restrict__ y, long cols, float eps) {
  const float* xr = x + (long)blockIdx.x * cols;
  float* yr = y + (long)blockIdx.x * cols;

  float sum = 0.0f;
  for (long i = threadIdx.x; i < cols; i += BLOCK) sum += xr[i];
  const float mean = block_sum<BLOCK>(sum) / cols;

  float var_acc = 0.0f;
  for (long i = threadIdx.x; i < cols; i += BLOCK) {
    const float d = xr[i] - mean;
    var_acc += d * d;
  }
  const float inv_std = rsqrtf(block_sum<BLOCK>(var_acc) / cols + eps);

  for (long i = threadIdx.x; i < cols; i += BLOCK) {
    yr[i] = (xr[i] - mean) * inv_std * w[i] + b[i];
  }
}

// fp16：所有中间量 fp32
template <int BLOCK>
__global__ void layer_norm_fp16_kernel(const __half* __restrict__ x,
                                       const float* __restrict__ w,
                                       const float* __restrict__ b,
                                       __half* __restrict__ y, long cols, float eps) {
  const __half* xr = x + (long)blockIdx.x * cols;
  __half* yr = y + (long)blockIdx.x * cols;

  float sum = 0.0f;
  for (long i = threadIdx.x; i < cols; i += BLOCK) sum += __half2float(xr[i]);
  const float mean = block_sum<BLOCK>(sum) / cols;

  float var_acc = 0.0f;
  for (long i = threadIdx.x; i < cols; i += BLOCK) {
    const float d = __half2float(xr[i]) - mean;
    var_acc += d * d;
  }
  const float inv_std = rsqrtf(block_sum<BLOCK>(var_acc) / cols + eps);

  for (long i = threadIdx.x; i < cols; i += BLOCK) {
    yr[i] = __float2half((__half2float(xr[i]) - mean) * inv_std * w[i] + b[i]);
  }
}

torch::Tensor layer_norm(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "x/w/b must be CUDA tensors");
  TORCH_CHECK(w.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat, "w/b must be fp32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, H]");
  const bool is_fp32 = x.scalar_type() == at::kFloat;
  const bool is_fp16 = x.scalar_type() == at::Half;
  TORCH_CHECK(is_fp32 || is_fp16, "fp32 / fp16 supported");
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto bc = b.contiguous();
  const long rows = xc.size(0), cols = xc.size(1);
  auto y = torch::empty_like(xc);
  if (rows == 0 || cols == 0) return y;

  int block = 64;
  while (block < cols && block < 1024) block <<= 1;

#define DISPATCH_LN(B)                                                         \
  do {                                                                         \
    if (is_fp32)                                                               \
      layer_norm_fp32_kernel<B><<<(unsigned int)rows, B>>>(                    \
          xc.data_ptr<float>(), wc.data_ptr<float>(), bc.data_ptr<float>(),    \
          y.data_ptr<float>(), cols, (float)eps);                              \
    else                                                                       \
      layer_norm_fp16_kernel<B><<<(unsigned int)rows, B>>>(                    \
          reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),            \
          wc.data_ptr<float>(), bc.data_ptr<float>(),                          \
          reinterpret_cast<__half*>(y.data_ptr<at::Half>()), cols, (float)eps);\
  } while (0)

  switch (block) {
    case 64: DISPATCH_LN(64); break;
    case 128: DISPATCH_LN(128); break;
    case 256: DISPATCH_LN(256); break;
    case 512: DISPATCH_LN(512); break;
    default: DISPATCH_LN(1024); break;
  }
#undef DISPATCH_LN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layer_norm", &layer_norm, "LayerNorm (CUDA)", py::arg("x"), py::arg("w"),
        py::arg("b"), py::arg("eps") = 1e-5);
}
