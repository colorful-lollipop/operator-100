"""006 Softmax —— 正确性测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import pytest
import torch

from common.ops100.testing import assert_close, load_problem

_HERE = Path(__file__).resolve().parent
_MODS = load_problem(_HERE, "006")

CASES = [
    (1, 64),
    (7, 256),
    (37, 5120),   # 非整块边界
    (128, 4096),
]
DTYPES = [(torch.float32, 1e-5), (torch.float16, 1e-3)]


@pytest.mark.parametrize("rows,cols", CASES)
@pytest.mark.parametrize("dtype,tol", DTYPES)
@pytest.mark.parametrize("name", sorted(_MODS))
def test_softmax(name, rows, cols, dtype, tol):
    if name != "reference" and not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    ref_fn = _MODS["reference"].softmax
    impl_fn = getattr(_MODS[name], "softmax")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = (torch.randn(rows, cols, device=dev) * 16).to(dtype)  # 大动态范围，考验 safe softmax
    got = impl_fn(x).float()
    want = ref_fn(x).float()
    assert_close(got, want, name=name, atol=tol, rtol=tol)
    # 行和恒等于 1 的不变量
    assert torch.allclose(got.sum(-1), torch.ones(rows, device=dev), atol=1e-3), \
        f"[{name}] 每行和不为 1"


def test_unsafe_softmax_overflow():
    """选做思考题的验证：不减 max 时 exp 上溢，safe 版本安然无恙。"""
    if "cuda" not in _MODS:
        pytest.skip("CUDA 后端不可用")
    x = torch.full((1, 1024), 100.0, device="cuda")  # e^100 溢出 fp32
    y = _MODS["cuda"].softmax(x)
    assert torch.isfinite(y).all(), "safe softmax 输出不应包含 inf/NaN"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
