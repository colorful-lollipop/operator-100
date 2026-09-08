"""005 矩阵转置 —— PyTorch 参考实现。"""

import torch


def transpose(x: torch.Tensor) -> torch.Tensor:
    """[rows, cols] -> [cols, rows]。"""
    return x.t().contiguous()


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.arange(12, device=dev, dtype=torch.float32).view(3, 4)
    print(transpose(x))
