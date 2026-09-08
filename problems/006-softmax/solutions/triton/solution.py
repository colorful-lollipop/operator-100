"""006 Softmax —— Triton 方案实现（一个 program 一行）。

局限：BLOCK 需一次性容纳整行（cols ≤ 16384）；更长的行请看题 031 online-softmax。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    # other=-inf：被掩码位置 exp(-inf - m) = 0，不污染分母
    x = tl.load(x_ptr + row * cols + offsets, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * cols + offsets, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "x 必须是二维 CUDA 张量"
    rows, cols = x.shape
    assert cols <= 16384, "cols 过大，请用题 031 的 online-softmax 方案"
    y = torch.empty_like(x)
    block = max(triton.next_power_of_2(cols), 16)
    num_warps = 4 if block <= 1024 else 8
    _softmax_kernel[(rows,)](x, y, cols, BLOCK=block, num_warps=num_warps)
    return y
