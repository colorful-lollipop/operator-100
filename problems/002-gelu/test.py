"""002 GELU —— 正确性测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import pytest
import torch

from common.ops100.testing import assert_close, load_problem

_HERE = Path(__file__).resolve().parent
_MODS = load_problem(_HERE, "002")

SHAPES = [
    (1,),
    (1024,),
    (255 * 1024 + 7,),
    (1 << 20,),
]
MODES = [(0, "erf", 1e-6), (1, "tanh", 1e-5)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode,tol_name,tol", MODES)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_gelu(name, mode, tol_name, tol, shape):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    ref_fn = _MODS["reference"].gelu
    impl_fn = getattr(_MODS[name], "gelu")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(shape, device=dev)
    assert_close(impl_fn(x, mode), ref_fn(x, mode), name=f"{name}:{tol_name}", atol=tol, rtol=tol)


def test_two_modes_differ_by_little():
    """erf 版与 tanh 版最大差 ~3e-4：这正是"精度换速度"的量化。"""
    if "cuda" not in _MODS:
        pytest.skip("CUDA 后端不可用")
    x = torch.randn(1 << 20, device="cuda")
    diff = (_MODS["cuda"].gelu(x, 0) - _MODS["cuda"].gelu(x, 1)).abs().max().item()
    assert diff < 1e-3, f"erf/tanh 版差异过大: {diff}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
