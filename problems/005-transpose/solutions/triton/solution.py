"""005 矩阵转置 —— Triton 方案实现（二维分块 + tl.trans）。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(x_ptr, y_ptr, rows, cols, BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rs = pid_r * BLOCK + tl.arange(0, BLOCK)  # 源行号
    cs = pid_c * BLOCK + tl.arange(0, BLOCK)  # 源列号

    # 读 x[rs, cs]（块内列方向连续 => 合并），掩码保护边界
    x = tl.load(x_ptr + rs[:, None] * cols + cs[None, :],
                mask=(rs[:, None] < rows) & (cs[None, :] < cols), other=0.0)
    # 写 y[cs, rs]（tl.trans 后行方向连续）
    tl.store(y_ptr + cs[:, None] * rows + rs[None, :], tl.trans(x),
             mask=(cs[:, None] < cols) & (rs[None, :] < rows))


def transpose(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "x 必须是二维 CUDA 张量"
    rows, cols = x.shape
    y = torch.empty(cols, rows, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK"]), triton.cdiv(cols, meta["BLOCK"]))
    _transpose_kernel[grid](x, y, rows, cols, BLOCK=32)
    return y
