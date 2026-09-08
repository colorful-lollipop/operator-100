"""004 归约求和 —— Triton 方案实现。

Triton 的两级归约：kernel1 每个 program 用 CHUNK 轮跨步累加写出部分和，
第二级直接用 torch.sum 演示（与 CUDA 版「单 block 归并」思想一致，工程上更省事）。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * CHUNK
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(CHUNK):
        offsets = base + k * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "输入必须是 CUDA 张量"
    BLOCK, CHUNK = 1024, 8
    n = x.numel()
    if n == 0:
        return x.new_zeros(())
    blocks = triton.cdiv(n, BLOCK * CHUNK)
    partial = torch.empty(blocks, device=x.device, dtype=torch.float32)
    _partial_sum_kernel[(blocks,)](x.view(-1), partial, n, BLOCK=BLOCK, CHUNK=CHUNK,
                                   num_warps=8)
    return partial.sum()
