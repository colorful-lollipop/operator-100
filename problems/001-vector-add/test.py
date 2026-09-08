"""001 向量加法 —— 正确性测试。

用法：python test.py（直接运行）或仓库根目录 pytest（自动收集）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import pytest
import torch

from common.ops100.testing import assert_close, load_problem

_HERE = Path(__file__).resolve().parent
_MODS = load_problem(_HERE, "001")

# 形状覆盖：最小规模、整块、非整块边界（考验边界保护）、大规模
SHAPES = [
    (1,),
    (256,),
    (255 * 1024 + 7,),
    (1 << 22,),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_vector_add(name, shape):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    ref_fn = _MODS["reference"].vector_add
    impl_fn = getattr(_MODS[name], "vector_add")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(shape, device=dev)
    b = torch.randn(shape, device=dev)
    assert_close(impl_fn(a, b), ref_fn(a, b), name=name, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", SHAPES[1:])
def test_vector_add_cuda_naive_vs_grid_stride(shape):
    """选做对照：naive 与 grid-stride 结果必须一致。"""
    if "cuda" not in _MODS:
        pytest.skip("CUDA 后端不可用")
    a = torch.randn(shape, device="cuda")
    b = torch.randn(shape, device="cuda")
    assert_close(
        _MODS["cuda"].vector_add(a, b, mode=0),
        _MODS["cuda"].vector_add(a, b, mode=1),
        name="cuda:naive-vs-stride",
        atol=0.0,
        rtol=0.0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
