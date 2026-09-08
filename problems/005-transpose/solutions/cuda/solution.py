"""005 矩阵转置 —— CUDA 方案入口。首次调用触发 JIT 编译，之后走缓存。"""

import os

import torch
from torch.utils.cpp_extension import load

_mod = load(
    name="ops100_005_transpose",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.cu")],
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],  # VS2026 等过新宿主编译器需放宽版本检查，常规环境无害
    verbose=False,
)


def transpose(x: torch.Tensor, mode: int = 1) -> torch.Tensor:
    return _mod.transpose(x, mode)
