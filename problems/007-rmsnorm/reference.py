"""007 RMSNorm —— PyTorch 参考实现。"""

import torch


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """x: [rows, H]，w: [H]。fp32 中间计算，输出 dtype 与 x 一致。"""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * w.float()).to(x.dtype)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 16, device=dev)
    w = torch.ones(16, device=dev)
    print(rms_norm(x, w))
