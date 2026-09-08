"""002 GELU —— 性能基准。memory-bound：bytes = 2 * n * 4（读 x、写 y）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根目录

import torch

from common.ops100.bench import report, time_ms
from common.ops100.testing import load_problem

N = 1 << 24


def main():
    if not torch.cuda.is_available():
        print("需要 NVIDIA GPU")
        return
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | N = {N}")
    mods = load_problem(Path(__file__).resolve().parent, "002")
    x = torch.randn(N, device="cuda")
    nbytes = 2 * N * 4

    for mode, tag in [(0, "erf"), (1, "tanh")]:
        print(f"-- mode={mode} ({tag}) --")
        report("torch eager", time_ms(lambda: mods["reference"].gelu(x, mode)), nbytes)
        if "cuda" in mods:
            report("cuda", time_ms(lambda: mods["cuda"].gelu(x, mode)), nbytes)
        if "triton" in mods:
            report("triton", time_ms(lambda: mods["triton"].gelu(x, mode)), nbytes)


if __name__ == "__main__":
    main()
