"""003 激活全家桶 —— PyTorch 参考实现（正确性与性能对照基准）。"""

import torch
import torch.nn.functional as F

ACT_RELU, ACT_SILU, ACT_SIGMOID = 0, 1, 2


def act(x: torch.Tensor, kind: int) -> torch.Tensor:
    """kind: 0=relu, 1=silu, 2=sigmoid"""
    if kind == ACT_RELU:
        return F.relu(x)
    if kind == ACT_SILU:
        return F.silu(x)
    if kind == ACT_SIGMOID:
        return F.sigmoid(x)
    raise ValueError(f"未知激活编号: {kind}")


def fused_bias_act(x: torch.Tensor, bias: torch.Tensor, kind: int) -> torch.Tensor:
    """y[i,j] = act(x[i,j] + bias[j])，x: [rows, cols], bias: [cols]。"""
    return act(x + bias, kind)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 8, device=dev)
    b = torch.randn(8, device=dev)
    print("silu :", act(x, 1))
    print("fused:", fused_bias_act(x, b, 1))
