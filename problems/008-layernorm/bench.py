"""008 LayerNorm —— 性能基准。

memory-bound 口径：bytes ≈ 3 * rows * H * 4（x 逻辑读两遍 + 写一遍）。
注意：每个 block 负责一行的第二次读几乎必然命中 L2（一行 fp32 仅 16KB），
所以实测 GB/s 可能超过 DRAM 峰值带宽——这不是 bug，是缓存的作用，
引导学生用 ncu 对比 DRAM 与 L2 吞吐即可看穿。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import torch

from common.ops100.bench import report, time_ms
from common.ops100.testing import load_problem

ROWS, COLS = 8192, 4096


def main():
    if not torch.cuda.is_available():
        print("需要 NVIDIA GPU")
        return
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {ROWS}x{COLS}")
    mods = load_problem(Path(__file__).resolve().parent, "008")

    for dtype in (torch.float32, torch.float16):
        x = torch.randn(ROWS, COLS, device="cuda", dtype=dtype)
        w = torch.randn(COLS, device="cuda")
        b = torch.randn(COLS, device="cuda")
        nbytes = 3 * ROWS * COLS * x.element_size()
        print(f"-- {dtype} --")
        report("torch eager", time_ms(lambda: mods["reference"].layer_norm(x, w, b)), nbytes)
        if "cuda" in mods:
            report("cuda", time_ms(lambda: mods["cuda"].layer_norm(x, w, b)), nbytes)
        if "triton" in mods:
            report("triton", time_ms(lambda: mods["triton"].layer_norm(x, w, b)), nbytes)


if __name__ == "__main__":
    main()
