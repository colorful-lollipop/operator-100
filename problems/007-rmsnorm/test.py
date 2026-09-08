"""007 RMSNorm —— 正确性测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import pytest
import torch

from common.ops100.testing import assert_close, load_problem

_HERE = Path(__file__).resolve().parent
_MODS = load_problem(_HERE, "007")

CASES = [
    (1, 64),
    (37, 5120),   # 非整块边界
    (128, 4096),  # LLaMA-7B hidden size
]
DTYPES = [(torch.float32, 1e-5), (torch.float16, 1e-2)]


@pytest.mark.parametrize("rows,cols", CASES)
@pytest.mark.parametrize("dtype,tol", DTYPES)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_rms_norm(name, rows, cols, dtype, tol):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    ref_fn = _MODS["reference"].rms_norm
    impl_fn = getattr(_MODS[name], "rms_norm")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = (torch.randn(rows, cols, device=dev) * 4).to(dtype)
    w = torch.randn(cols, device=dev)
    assert_close(impl_fn(x, w), ref_fn(x, w), name=name, atol=tol, rtol=tol)


def test_zero_row_no_nan():
    """全零行依赖 eps 兜底，不应产生 NaN。"""
    if "cuda" not in _MODS:
        pytest.skip("CUDA 后端不可用")
    x = torch.zeros(2, 1024, device="cuda")
    w = torch.ones(1024, device="cuda")
    y = _MODS["cuda"].rms_norm(x, w)
    assert torch.isfinite(y).all(), "全零行不应产生 NaN（检查 eps 位置）"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
