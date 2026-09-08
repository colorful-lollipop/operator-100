"""001 向量加法 —— Triton 方案实现。需要 triton >= 3.0（Windows 可用 triton-windows 轮子）。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """每个 program 处理 BLOCK_SIZE 个元素，掩码保护尾部。"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.is_cuda, "输入必须为同形状 CUDA 张量"
    c = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024, num_warps=4)
    return c
