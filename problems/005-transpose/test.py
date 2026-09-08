"""005 矩阵转置 —— 正确性测试。转置是精确操作：容差为 0。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import pytest
import torch

from common.ops100.testing import assert_close, load_problem

_HERE = Path(__file__).resolve().parent
_MODS = load_problem(_HERE, "005")

SHAPES = [
    (1, 1),
    (1, 100),
    (64, 64),
    (33, 65),   # 非整块边界
    (512, 4096),
    (4096, 512),
]


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_transpose(name, rows, cols):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    ref_fn = _MODS["reference"].transpose
    kwargs = {"mode": 0} if name == "cuda" else {}
    impl_fn = getattr(_MODS[name], "transpose")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(rows, cols, device=dev)
    assert_close(impl_fn(x, **kwargs), ref_fn(x), name=name, atol=0.0, rtol=0.0)


def test_cuda_tiled_vs_naive_exact():
    if "cuda" not in _MODS:
        pytest.skip("CUDA 后端不可用")
    x = torch.randn(512, 4096, device="cuda")
    assert_close(
        _MODS["cuda"].transpose(x, mode=0),
        _MODS["cuda"].transpose(x, mode=1),
        name="cuda:naive-vs-tiled", atol=0.0, rtol=0.0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
