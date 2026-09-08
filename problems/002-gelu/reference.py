"""002 GELU —— PyTorch 参考实现（正确性与性能对照基准）。"""

import torch
import torch.nn.functional as F


def gelu(x: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """mode: 0=erf 精确版, 1=tanh 近似版。"""
    if mode == 0:
        return F.gelu(x)
    return F.gelu(x, approximate="tanh")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(8, device=dev)
    print("erf :", gelu(x, 0))
    print("tanh:", gelu(x, 1))
