// 001 向量加法 —— CUDA 实现
// 两个版本：naive（一线程一元素）与 grid-stride（线程数与规模解耦）。
// 编译由 solutions/cuda/solution.py 通过 torch.utils.cpp_extension.load 自动完成。
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// 版本 A：一个线程处理一个元素。n 不是 block 倍数时必须做边界保护。
__global__ void vector_add_naive(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// 版本 B：grid-stride loop。固定网格规模，循环推进直到覆盖全部元素，
// 是生产代码的推荐写法（规模变化时无需重新划分网格）。
__global__ void vector_add_grid_stride(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c, long n) {
  long stride = (long)gridDim.x * blockDim.x;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b, int64_t mode) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
              "only fp32 is supported (fp16/bf16 is optional)");
  TORCH_CHECK(a.sizes() == b.sizes(), "a/b must have the same shape");
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = torch::empty_like(a_c);
  const long n = a_c.numel();
  if (n == 0) return c;

  const int block = 256;
  // 故意限制网格大小，迫使 grid-stride 版本走多轮循环（演示用）
  const long grid_cap = 65535L;
  if (mode == 0) {
    const long grid = (n + block - 1) / block;
    vector_add_naive<<<(unsigned int)grid, block>>>(
        a_c.data_ptr<float>(), b_c.data_ptr<float>(), c.data_ptr<float>(), n);
  } else {
    const long grid = std::min((n + block - 1) / block, grid_cap);
    vector_add_grid_stride<<<(unsigned int)grid, block>>>(
        a_c.data_ptr<float>(), b_c.data_ptr<float>(), c.data_ptr<float>(), n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "vector add (CUDA)", py::arg("a"), py::arg("b"),
        py::arg("mode") = 1);  // mode: 0=naive, 1=grid-stride
}
