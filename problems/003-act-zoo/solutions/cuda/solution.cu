// 003 激活全家桶 —— CUDA 实现
// 教学点：这里用「运行期 int 分支」统一三种激活；分支对 warp 内所有线程一致，
// 不会产生发散，实测与模板分发性能几乎一致（选做题让你亲手验证）。
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float apply_act(float v, int kind) {
  if (kind == 0) return fmaxf(v, 0.0f);        // relu
  if (kind == 1) return v / (1.0f + expf(-v)); // silu
  return 1.0f / (1.0f + expf(-v));             // sigmoid（e^{-x} 上溢时结果为 0，安全）
}

__global__ void act_kernel(const float* __restrict__ x, float* __restrict__ y,
                           long n, int kind) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = apply_act(x[i], kind);
}

// x: [rows, cols] row-major，bias: [cols]。y[i,j] = act(x[i,j] + bias[j])
__global__ void fused_bias_act_kernel(const float* __restrict__ x,
                                      const float* __restrict__ bias,
                                      float* __restrict__ y, long rows,
                                      long cols, int kind) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    long col = idx % cols;
    y[idx] = apply_act(x[idx] + bias[col], kind);
  }
}

static void check_common(const torch::Tensor& x, int64_t kind) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "only fp32 is supported");
  TORCH_CHECK(kind >= 0 && kind <= 2, "kind must be 0 (relu), 1 (silu) or 2 (sigmoid)");
}

torch::Tensor act(torch::Tensor x, int64_t kind) {
  check_common(x, kind);
  auto xc = x.contiguous();
  auto y = torch::empty_like(xc);
  const long n = xc.numel();
  if (n == 0) return y;
  const int block = 256;
  const long grid = (n + block - 1) / block;
  act_kernel<<<(unsigned int)grid, block>>>(xc.data_ptr<float>(),
                                            y.data_ptr<float>(), n, (int)kind);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor fused_bias_act(torch::Tensor x, torch::Tensor bias, int64_t kind) {
  check_common(x, kind);
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat, "bias must be a fp32 CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, cols]");
  TORCH_CHECK(bias.numel() == x.size(1), "bias length must equal cols");
  auto xc = x.contiguous();
  auto bc = bias.contiguous();
  auto y = torch::empty_like(xc);
  const long rows = xc.size(0), cols = xc.size(1);
  if (rows == 0 || cols == 0) return y;
  const int block = 256;
  const long grid = (rows * cols + block - 1) / block;
  fused_bias_act_kernel<<<(unsigned int)grid, block>>>(
      xc.data_ptr<float>(), bc.data_ptr<float>(), y.data_ptr<float>(), rows, cols,
      (int)kind);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("act", &act, "activation (CUDA)", py::arg("x"), py::arg("kind"));
  m.def("fused_bias_act", &fused_bias_act, "fused bias+act (CUDA)",
        py::arg("x"), py::arg("bias"), py::arg("kind"));
}
