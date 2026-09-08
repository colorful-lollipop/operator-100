"""008 LayerNorm —— PyTorch 参考实现。"""

import torch


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """x: [rows, H]，w/b: [H]。fp32 中间计算，输出 dtype 与 x 一致。"""
    return torch.nn.functional.layer_norm(x.float(), (x.size(-1),), w.float(), b.float(), eps).to(x.dtype)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 16, device=dev)
    w = torch.ones(16, device=dev)
    b = torch.zeros(16, device=dev)
    print(layer_norm(x, w, b))
