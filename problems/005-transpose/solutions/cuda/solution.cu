// 005 矩阵转置 —— CUDA 实现（naive + shared memory tiled）
// 教学点：naive 版读合并、写分散；tiled 版让读写两头都合并；
// [TILE][TILE+1] 的 padding 消除 shared memory bank conflict。
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

constexpr int kTile = 32;

// 朴素版：读 x[r][c] 合并（c 连续），写 y[c][r] 分散（步长 rows）
__global__ void transpose_naive_kernel(const float* __restrict__ x,
                                       float* __restrict__ y, int rows, int cols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) y[(long)c * rows + r] = x[(long)r * cols + c];
}

// tiled 版：经共享内存中转，读写两侧都合并访问
__global__ void transpose_tiled_kernel(const float* __restrict__ x,
                                       float* __restrict__ y, int rows, int cols) {
  // +1 padding：warp 按列读 tile[tx][ty] 时不再全落同一 bank
  __shared__ float tile[kTile][kTile + 1];

  const int c = blockIdx.x * kTile + threadIdx.x;  // 源列
  const int r = blockIdx.y * kTile + threadIdx.y;  // 源行
  if (r < rows && c < cols) tile[threadIdx.y][threadIdx.x] = x[(long)r * cols + c];
  __syncthreads();

  // 输出行 = 源列方向，输出列 = 源行方向；warp 内 tx 连续 => 输出地址连续 => 合并
  const int oc = blockIdx.y * kTile + threadIdx.x;  // 输出列（对应源行）
  const int orow = blockIdx.x * kTile + threadIdx.y;  // 输出行（对应源列）
  if (orow < cols && oc < rows) y[(long)orow * rows + oc] = tile[threadIdx.x][threadIdx.y];
}

torch::Tensor transpose(torch::Tensor x, int64_t mode) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "only fp32 is supported");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, cols]");
  TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (naive) or 1 (tiled)");
  auto xc = x.contiguous();
  const int rows = (int)xc.size(0), cols = (int)xc.size(1);
  auto y = torch::empty({cols, rows}, xc.options());
  if (rows == 0 || cols == 0) return y;

  dim3 block(kTile, kTile);
  dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  if (mode == 0) {
    transpose_naive_kernel<<<grid, block>>>(xc.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
  } else {
    transpose_tiled_kernel<<<grid, block>>>(xc.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose", &transpose, "transpose (CUDA)", py::arg("x"), py::arg("mode") = 1);
}
