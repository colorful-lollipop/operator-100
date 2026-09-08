"""004 归约求和 —— 性能基准。memory-bound：bytes ≈ n * 4（只读一次输入）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import torch

from common.ops100.bench import report, time_ms
from common.ops100.testing import load_problem

N = 1 << 26  # 64M 元素 = 256 MB，归约要有足够数据量才看得出差距


def main():
    if not torch.cuda.is_available():
        print("需要 NVIDIA GPU")
        return
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | N = {N}")
    mods = load_problem(Path(__file__).resolve().parent, "004")
    x = torch.randn(N, device="cuda")
    nbytes = N * 4

    report("torch eager", time_ms(lambda: mods["reference"].reduce_sum(x)), nbytes)
    if "cuda" in mods:
        report("cuda (两级归约)", time_ms(lambda: mods["cuda"].reduce_sum(x)), nbytes)
    if "triton" in mods:
        report("triton", time_ms(lambda: mods["triton"].reduce_sum(x)), nbytes)


if __name__ == "__main__":
    main()
