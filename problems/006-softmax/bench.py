"""006 Softmax —— 性能基准。memory-bound：fp32 bytes = 2 * rows * cols * 4。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import torch

from common.ops100.bench import report, time_ms
from common.ops100.testing import load_problem

ROWS, COLS = 8192, 4096  # 典型场景：注意力 score 矩阵的 softmax


def main():
    if not torch.cuda.is_available():
        print("需要 NVIDIA GPU")
        return
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {ROWS}x{COLS}")
    mods = load_problem(Path(__file__).resolve().parent, "006")

    for dtype in (torch.float32, torch.float16):
        x = (torch.randn(ROWS, COLS, device="cuda") * 8).to(dtype)
        nbytes = 2 * ROWS * COLS * x.element_size()
        print(f"-- {dtype} --")
        report("torch eager", time_ms(lambda: mods["reference"].softmax(x)), nbytes)
        if "cuda" in mods:
            report("cuda", time_ms(lambda: mods["cuda"].softmax(x)), nbytes)
        if "triton" in mods:
            report("triton", time_ms(lambda: mods["triton"].softmax(x)), nbytes)


if __name__ == "__main__":
    main()
