"""operator-100 共用基准计时工具。

计时口径统一为 CUDA Event（GPU 张量）/ perf_counter（CPU 张量），
先 warmup 再多次迭代取平均；上报带宽或算力时请由题目提供
bytes / flops 计算公式（见 docs/benchmark.md）。
"""

from __future__ import annotations

import time

import torch


def time_ms(fn, warmup: int = 10, iters: int = 100) -> float:
    """返回 fn() 单次调用平均耗时（毫秒）。"""
    if not torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1e3 / iters

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def gb_s(nbytes: float, ms: float) -> float:
    """有效带宽 GB/s。nbytes 为一次调用读+写的总字节数。"""
    return nbytes / 1e9 / (ms / 1e3)


def tflops(flops: float, ms: float) -> float:
    """有效算力 TFLOPS。"""
    return flops / 1e12 / (ms / 1e3)


def report(title: str, ms: float, nbytes: float | None = None, flops: float | None = None) -> None:
    """打印一行基准结果：耗时（必填）+ 带宽 / 算力（可选）。"""
    line = f"{title:<28} {ms:9.3f} ms"
    if nbytes:
        line += f" | {gb_s(nbytes, ms):8.1f} GB/s"
    if flops:
        line += f" | {tflops(flops, ms):7.2f} TFLOPS"
    print(line)
