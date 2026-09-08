"""001 向量加法 —— 性能基准。

memory-bound：bytes = 3 * n * 4（读 a、读 b、写 c 各 4 字节/元素）。
RTX 4070 Ti SUPER 理论带宽 ~672 GB/s，看各实现能打多少。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import torch

from common.ops100.bench import report, time_ms
from common.ops100.testing import load_problem

N = 1 << 24  # 16M 元素 = 64 MB/数组


def main():
    if not torch.cuda.is_available():
        print("需要 NVIDIA GPU")
        return
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | N = {N}")
    mods = load_problem(Path(__file__).resolve().parent, "001")

    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    nbytes = 3 * N * 4  # 读 a + 读 b + 写 c

    report("torch eager", time_ms(lambda: mods["reference"].vector_add(a, b)), nbytes)
    if "cuda" in mods:
        ms_stride = time_ms(lambda: mods["cuda"].vector_add(a, b, mode=1))
        report("cuda (grid-stride)", ms_stride, nbytes)
        report("cuda (naive)     ", time_ms(lambda: mods["cuda"].vector_add(a, b, mode=0)), nbytes)
    if "triton" in mods:
        report("triton", time_ms(lambda: mods["triton"].vector_add(a, b)), nbytes)


if __name__ == "__main__":
    main()
