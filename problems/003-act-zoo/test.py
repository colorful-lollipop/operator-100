"""003 激活全家桶 —— 正确性测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import pytest
import torch

from common.ops100.testing import assert_close, load_problem

_HERE = Path(__file__).resolve().parent
_MODS = load_problem(_HERE, "003")

SHAPES = [(1,), (1024,), (255 * 1024 + 7,)]
FUSED_SHAPES = [(1, 64), (37, 5120), (128, 4096)]
KINDS = [0, 1, 2]  # relu / silu / sigmoid


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_act(name, kind, shape):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    ref_fn = _MODS["reference"].act
    impl_fn = getattr(_MODS[name], "act")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(shape, device=dev) * 4  # 拉大动态范围，覆盖大 |x| 区域
    assert_close(impl_fn(x, kind), ref_fn(x, kind), name=name, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("rows,cols", FUSED_SHAPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_fused_bias_act(name, kind, rows, cols):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    ref_fn = _MODS["reference"].fused_bias_act
    impl_fn = getattr(_MODS[name], "fused_bias_act")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(rows, cols, device=dev)
    bias = torch.randn(cols, device=dev)
    assert_close(impl_fn(x, bias, kind), ref_fn(x, bias, kind), name=name, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
