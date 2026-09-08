"""005 矩阵转置 —— CUDA 方案入口。首次调用触发 JIT 编译，之后走缓存。"""

import os

import torch
from torch.utils.cpp_extension import load

_mod = load(
    name="ops100_005_transpose",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.cu")],
    # -allow-unsupported-compiler：VS2026 等过新宿主编译器需放宽版本检查
    # -Xcompiler=/Zc:preprocessor：CUDA 13.x 的 CCCL 头要求 MSVC 启用标准预处理器（仅 Windows）
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"]
    + (["-Xcompiler=/Zc:preprocessor"] if os.name == "nt" else []),
    verbose=False,
)


def transpose(x: torch.Tensor, mode: int = 1) -> torch.Tensor:
    """mode: 0=naive，1=tiled（默认，shared memory 分块）。"""
    return _mod.transpose(x, mode)
