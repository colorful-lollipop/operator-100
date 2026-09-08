"""008 LayerNorm —— CUDA 方案入口。首次调用触发 JIT 编译，之后走缓存。"""

import os

import torch
from torch.utils.cpp_extension import load

_mod = load(
    name="ops100_008_layernorm",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.cu")],
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],  # VS2026 等过新宿主编译器需放宽版本检查，常规环境无害
    verbose=False,
)


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return _mod.layer_norm(x, w, b, eps)
