"""004 归约求和 —— 正确性测试。

关键点：fp32 求和的「真值」必须用 fp64 计算——任何两个 fp32 实现（包括 torch
与你的 kernel）只是求和顺序不同，相互比较会误判。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import pytest
import torch

from common.ops100.testing import assert_close, load_problem

_HERE = Path(__file__).resolve().parent
_MODS = load_problem(_HERE, "004")

SHAPES = [
    (0,),            # 空张量
    (1,),
    (256 * 8,),
    (255 * 2048 + 7,),  # 非整段边界
    (1 << 22,),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_reduce_sum_vs_fp64_truth(name, shape):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    impl_fn = getattr(_MODS[name], "reduce_sum")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(shape, device=dev)
    truth = x.double().sum()  # fp64 真值（CPU/GPU 皆可）
    # fp32 求和 n=4M 时的误差量级：rtol 1e-3 / atol 1e-2 是安全的
    assert_close(impl_fn(x), truth, name=name, atol=1e-2, rtol=1e-3)


def test_determinism_note():
    """两级归约对固定输入是确定的（同一 kernel、同一划分）；验证同输入两次结果一致。"""
    if "cuda" not in _MODS:
        pytest.skip("CUDA 后端不可用")
    x = torch.randn(1 << 20, device="cuda")
    a = _MODS["cuda"].reduce_sum(x).item()
    b = _MODS["cuda"].reduce_sum(x).item()
    assert a == b, "同一输入两次运行结果不一致（不应发生：本实现无原子竞争）"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
