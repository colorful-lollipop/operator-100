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
    bias = torch.randn(N, device="cuda")  # fused 场景退化为无广播，测纯 elementwise 上限
    nbytes = 2 * N * 4

    for kind, tag in [(0, "relu"), (1, "silu"), (2, "sigmoid")]:
        print(f"-- {tag} --")
        report("torch eager", time_ms(lambda: mods["reference"].act(x, kind)), nbytes)
        if "cuda" in mods:
            report("cuda", time_ms(lambda: mods["cuda"].act(x, kind)), nbytes)
        if "triton" in mods:
            report("triton", time_ms(lambda: mods["triton"].act(x, kind)), nbytes)


if __name__ == "__main__":
    main()
