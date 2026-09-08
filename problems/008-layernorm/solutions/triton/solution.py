"""008 LayerNorm —— Triton 方案实现。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    x = tl.load(x_ptr + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / cols
    # 掩码位置 x=0 会让 (0-mean)^2 混入方差？——不会：mean 是有效均值，
    # 但为严谨起见用 tl.where 把无效位置清零
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / cols
    inv_std = tl.rsqrt(var + eps)

    y = (x - mean) * inv_std * w + b
    tl.store(y_ptr + row * cols + offsets, y, mask=mask)


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "x 必须是二维 CUDA 张量"
    rows, cols = x.shape
    assert w.numel() == cols and b.numel() == cols, "w/b 长度必须等于 H"
    y = torch.empty_like(x)
    block = max(triton.next_power_of_2(cols), 16)
    num_warps = 4 if block <= 1024 else 8
    _layer_norm_kernel[(rows,)](x, w, b, y, cols, eps, BLOCK=block, num_warps=num_warps)
    return y
