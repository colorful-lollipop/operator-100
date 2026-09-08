"""001 向量加法 —— PyTorch 参考实现（正确性与性能对照基准）。"""

import torch


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """c[i] = a[i] + b[i]"""
    return a + b


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(8, device=dev)
    b = torch.randn(8, device=dev)
    print("a + b =", vector_add(a, b))
