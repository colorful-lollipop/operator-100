"""006 Softmax —— PyTorch 参考实现。"""

import torch


def softmax(x: torch.Tensor) -> torch.Tensor:
    """沿最后一维的 safe softmax（torch 内部已处理数值稳定）。"""
    return torch.softmax(x, dim=-1)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 8, device=dev)
    print(softmax(x).sum(-1))  # 每行和应为 1
