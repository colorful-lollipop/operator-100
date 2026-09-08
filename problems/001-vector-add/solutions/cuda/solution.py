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
    # -allow-unsupported-compiler：VS2026 等过新宿主编译器需放宽版本检查
    # -Xcompiler=/Zc:preprocessor：CUDA 13.x 的 CCCL 头要求 MSVC 启用标准预处理器（仅 Windows）
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"]
    + (["-Xcompiler=/Zc:preprocessor"] if os.name == "nt" else []),
    verbose=False,
)


def vector_add(a: torch.Tensor, b: torch.Tensor, mode: int = 1) -> torch.Tensor:
    """mode: 0=naive（一线程一元素），1=grid-stride（默认）。"""
    return _mod.vector_add(a, b, mode)
