// 002 GELU —— CUDA 实现
// 精确版(erf)与近似版(tanh)共用一个 kernel，用模板在编译期选择，零运行期开销。
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// 精确版：0.5x(1 + erf(x/sqrt(2)))
__device__ __forceinline__ float gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
}

// 近似版：0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))，GPT 系列常用
__device__ __forceinline__ float gelu_tanh(float x) {
  const float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
  const float kKappa = 0.044715f;
  const float inner = kBeta * (x + kKappa * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

template <int MODE>
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y,
                            long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = MODE == 0 ? gelu_erf(x[i]) : gelu_tanh(x[i]);
}

torch::Tensor gelu(torch::Tensor x, int64_t mode) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "only fp32 is supported");
  TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (erf) or 1 (tanh)");
  auto xc = x.contiguous();
  auto y = torch::empty_like(xc);
  const long n = xc.numel();
  if (n == 0) return y;

  const int block = 256;
  const long grid = (n + block - 1) / block;
  if (mode == 0) {
    gelu_kernel<0><<<(unsigned int)grid, block>>>(xc.data_ptr<float>(),
                                                  y.data_ptr<float>(), n);
  } else {
    gelu_kernel<1><<<(unsigned int)grid, block>>>(xc.data_ptr<float>(),
                                                  y.data_ptr<float>(), n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu", &gelu, "GELU (CUDA)", py::arg("x"), py::arg("mode") = 0);
}
