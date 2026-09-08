"""001 向量加法 —— CUDA 方案入口。

首次调用会触发 JIT 编译（1~2 分钟，之后走缓存）。
需要：CUDA Toolkit（>=12）+ MSVC（Windows）或 gcc（Linux）。
"""

import os

import torch
from torch.utils.cpp_extension import load

_mod = load(
    name="ops100_001_vector_add",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.cu")],
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],  # VS2026 等过新宿主编译器需放宽版本检查，常规环境无害
    verbose=False,
)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _mod.vector_add(a, b)
