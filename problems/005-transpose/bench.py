"""005 矩阵转置 —— 性能基准。

memory-bound：bytes = 2 * rows * cols * 4。
期待：naive 写不合并（~一半带宽），tiled 接近峰值带宽。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import torch

from common.ops100.bench import report, time_ms
from common.ops100.testing import load_problem

ROWS, COLS = 8192, 8192


def main():
    if not torch.cuda.is_available():
        print("需要 NVIDIA GPU")
        return
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {ROWS}x{COLS}")
    mods = load_problem(Path(__file__).resolve().parent, "005")
    x = torch.randn(ROWS, COLS, device="cuda")
    nbytes = 2 * ROWS * COLS * 4

    report("torch eager", time_ms(lambda: mods["reference"].transpose(x)), nbytes)
    if "cuda" in mods:
        ms_naive = time_ms(lambda: mods["cuda"].transpose(x, mode=0))
        ms_tiled = time_ms(lambda: mods["cuda"].transpose(x, mode=1))
        report("cuda (naive) ", ms_naive, nbytes)
        report("cuda (tiled) ", ms_tiled, nbytes)
        print(f"tiled/naive 加速比: {ms_naive / ms_tiled:.2f}x")
    if "triton" in mods:
        report("triton", time_ms(lambda: mods["triton"].transpose(x)), nbytes)


if __name__ == "__main__":
    main()
