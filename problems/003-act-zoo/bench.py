"""003 激活全家桶 —— 性能基准。memory-bound：bytes = 2 * n * 4。"""

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
    mods = load_problem(Path(__file__).resolve().parent, "003")
    x = torch.randn(N, device="cuda")
    nbytes = 2 * N * 4

    for kind, tag in [(0, "relu"), (1, "silu"), (2, "sigmoid")]:
        print(f"-- {tag} --")
        report("torch eager", time_ms(lambda k=kind: mods["reference"].act(x, k)), nbytes)
        if "cuda" in mods:
            report("cuda", time_ms(lambda k=kind: mods["cuda"].act(x, k)), nbytes)
        if "triton" in mods:
            report("triton", time_ms(lambda k=kind: mods["triton"].act(x, k)), nbytes)


if __name__ == "__main__":
    main()
