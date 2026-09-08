"""008 LayerNorm —— 正确性测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import pytest
import torch

from common.ops100.testing import assert_close, load_problem

_HERE = Path(__file__).resolve().parent
_MODS = load_problem(_HERE, "008")

CASES = [
    (1, 64),
    (37, 5120),
    (128, 4096),
]
DTYPES = [(torch.float32, 1e-4), (torch.float16, 1e-2)]


@pytest.mark.parametrize("rows,cols", CASES)
@pytest.mark.parametrize("dtype,tol", DTYPES)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_layer_norm(name, rows, cols, dtype, tol):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    ref_fn = _MODS["reference"].layer_norm
    impl_fn = getattr(_MODS[name], "layer_norm")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = (torch.randn(rows, cols, device=dev) * 8).to(dtype)
    w = torch.randn(cols, device=dev)
    b = torch.randn(cols, device=dev)
    got = impl_fn(x, w, b).float()
    assert_close(got, ref_fn(x, w, b).float(), name=name, atol=tol, rtol=tol)
    # 归一化不变量（忽略 w/b 缩放前）：粗验每行标准化统计量
    xf = x.float()
    want_norm = (xf - xf.mean(-1, keepdim=True)) / torch.sqrt(xf.var(-1, keepdim=True, unbiased=False) + 1e-5)
    assert_close(got, want_norm * w.float() + b.float(), name=f"{name}:invariant", atol=tol * 2, rtol=tol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
