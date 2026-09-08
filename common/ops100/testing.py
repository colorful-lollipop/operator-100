"""operator-100 共用测试工具。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def has_cuda() -> bool:
    return torch.cuda.is_available()


def assert_close(actual, expected, *, name: str = "impl", atol: float = 1e-4, rtol: float = 1e-4):
    """带题目上下文的正确性断言（自动放宽 fp16/bf16 的 dtype 干扰，统一转 fp32 比较）。"""
    assert isinstance(actual, torch.Tensor), f"[{name}] 输出必须是 torch.Tensor，得到 {type(actual)}"
    assert actual.shape == expected.shape, f"[{name}] 形状不一致: {actual.shape} vs {expected.shape}"
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        msg=lambda m: f"[{name}] {m}",
    )


def load_module(unique_name: str, file_path: str | Path):
    """按文件路径加载模块，并注册为全局唯一名字。

    每道题都有同名的 reference.py / solutions/*/solution.py，
    pytest 一次性收集全部题目时直接 import 会发生模块名冲突，
    因此这里为每个模块分配 `ops100_<题号>_<后端>` 这样的唯一名字。
    """
    spec = importlib.util.spec_from_file_location(unique_name, str(file_path))
    assert spec is not None and spec.loader is not None, f"无法加载模块: {file_path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_problem(here: str | Path, pid: str) -> dict:
    """加载一道题的全部实现模块。

    here: 题目目录（如 problems/001-vector-add）；pid: 三位题号，如 "001"。
    返回 {"reference": <module>, "cuda": <module>?, "triton": <module>?}，
    缺少依赖（无 GPU / 未装 triton / 无编译器）的后端自动跳过并打印原因。
    """
    here = Path(here)
    mods = {"reference": load_module(f"ops100_{pid}_reference", here / "reference.py")}
    if not torch.cuda.is_available():
        return mods
    for backend in ("cuda", "triton"):
        path = here / "solutions" / backend / "solution.py"
        if not path.exists():
            continue
        try:
            mods[backend] = load_module(f"ops100_{pid}_{backend}", path)
        except Exception as e:  # noqa: BLE001 —— 编译器/依赖缺失时跳过该后端
            print(f"[skip] 题 {pid} 后端 {backend}: {type(e).__name__}: {e}")
    return mods
