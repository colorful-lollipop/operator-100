"""003 激活全家桶 —— Triton 方案实现。

教学点：kind 是 tl.constexpr，分支在编译期展开成三种独立 kernel 二进制——
与 CUDA 版的运行期分支形成对照。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _act_kernel(x_ptr, y_ptr, n, KIND: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if KIND == 0:  # relu
        y = tl.maximum(x, 0.0)
    else:
        s = tl.sigmoid(x)
        if KIND == 1:  # silu
            y = x * s
        else:  # sigmoid
            y = s
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _fused_bias_act_kernel(x_ptr, bias_ptr, y_ptr, rows, cols, KIND: tl.constexpr,
                           BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows * cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets % cols, mask=mask, other=0.0)
    v = x + bias
    if KIND == 0:
        y = tl.maximum(v, 0.0)
    else:
        s = tl.sigmoid(v)
        y = v * s if KIND == 1 else s
    tl.store(y_ptr + offsets, y, mask=mask)


def act(x: torch.Tensor, kind: int) -> torch.Tensor:
    assert x.is_cuda and 0 <= kind <= 2, "输入必须是 CUDA 张量，kind 取 0/1/2"
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: B023 —— grid 立即调用，闭包安全
    _act_kernel[grid](x, y, n, KIND=kind, BLOCK=1024, num_warps=4)
    return y


def fused_bias_act(x: torch.Tensor, bias: torch.Tensor, kind: int) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "x 必须是二维 CUDA 张量"
    assert bias.numel() == x.size(1), "bias 长度必须等于 cols"
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: B023 —— grid 立即调用，闭包安全
    _fused_bias_act_kernel[grid](x, bias, y, x.size(0), x.size(1), KIND=kind,
                                 BLOCK=1024, num_warps=4)
    return y
