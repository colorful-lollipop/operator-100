"""004 归约求和 —— PyTorch 参考实现（性能对照；正确性真值在 test.py 用 fp64 计算）。"""

import torch


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """全 tensor 求和，返回 0 维标量张量。"""
    return x.sum()


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1024, device=dev)
    print("sum =", reduce_sum(x).item())
