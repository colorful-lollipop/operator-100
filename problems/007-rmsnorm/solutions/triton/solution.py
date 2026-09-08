"""007 RMSNorm —— Triton 方案实现。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    x = tl.load(x_ptr + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ssq = tl.sum(x * x, axis=0) / cols
    inv = tl.rsqrt(ssq + eps)  # eps 在开方内、除法外
    y = x * inv * w
    tl.store(y_ptr + row * cols + offsets, y, mask=mask)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "x 必须是二维 CUDA 张量"
    rows, cols = x.shape
    assert w.numel() == cols, "w 长度必须等于 H"
    y = torch.empty_like(x)
    block = max(triton.next_power_of_2(cols), 16)
    num_warps = 4 if block <= 1024 else 8
    _rms_norm_kernel[(rows,)](x, w, y, cols, eps, BLOCK=block, num_warps=num_warps)
    return y
