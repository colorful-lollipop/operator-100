"""002 GELU —— Triton 方案实现。"""

import torch
import triton
import triton.language as tl

# triton >= 3.8 要求 jit 函数内引用的模块级常量必须是 tl.constexpr 实例
_SQRT1_2 = tl.constexpr(0.7071067811865476)
_BETA = tl.constexpr(0.7978845608028654)  # sqrt(2/pi)
_KAPPA = tl.constexpr(0.044715)


@triton.jit
def _gelu_kernel(x_ptr, y_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    """MODE 在编译期确定：0=erf 精确版，1=tanh 近似版。

    tanh 用恒等式 tanh(v) = 1 - 2/(exp(2v)+1) 手写：
    v 很大时 exp 溢出为 inf，2/inf -> 0，结果趋近 1，数学上仍然安全。
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if MODE == 0:
        y = 0.5 * x * (1.0 + tl.erf(x * _SQRT1_2))
    else:
        inner = _BETA * (x + _KAPPA * x * x * x)
        t = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
        y = 0.5 * x * (1.0 + t)
    tl.store(y_ptr + offsets, y, mask=mask)


def gelu(x: torch.Tensor, mode: int = 0) -> torch.Tensor:
    assert x.is_cuda, "输入必须是 CUDA 张量"
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _gelu_kernel[grid](x, y, n, MODE=mode, BLOCK=1024, num_warps=4)
    return y
