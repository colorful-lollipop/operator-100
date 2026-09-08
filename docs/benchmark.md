# 评测规范（Benchmark Rules）

> 本项目的性能题不是「跑得快就行」——**口径统一才可比**。

## 1. 计时

- 使用 `common.ops100.bench.time_ms`：CUDA Event 计时，warmup ≥ 10 次、迭代 ≥ 100 次取平均；
- 计时区间必须包含 kernel 全部 launch（多 kernel 方案天然公平）；
- 禁止在计时循环里做 `torch.empty` 分配之外的同步（如每步 `synchronize` 会把 launch 开销算进去，除非题目本身就是考 launch 开销，如 088）。

## 2. 带宽 / 算力口径（roofline）

每个 `bench.py` 必须在注释里写明 bytes / flops 公式，并打印其中适用的一个：

- **memory-bound**（逐元素、归约、转置、softmax、GEMV…）：
  `bytes = Σ(每个输入张量 nbytes + 每个输出张量 nbytes)`，报告 `GB/s = bytes / ms`；
- **compute-bound**（GEMM、attention、卷积…）：
  `flops = 2·M·N·K`（GEMM）、`2·N·d_model·d_head`（attention 前向近似），报告 `TFLOPS = flops / ms`。

判断口诀：算术强度 FLOPs/Bytes 低于设备 ridge point（RTX 40 系约 ~200 FLOP/B 量级）的算子按带宽评。

## 3. 正确性容差

| dtype | atol | rtol | 说明 |
|---|---|---|---|
| fp32 | 1e-4 | 1e-4 | 归约类（004/034）放宽至 rtol 1e-3，因求和顺序不同 |
| fp16 | 1e-2 | 1e-2 | softmax/norm 类输出 ≤1，可再收紧到 1e-3，见各题 test.py |
| bf16 | 2e-2 | 2e-2 | |

- 比较统一走 `common.ops100.testing.assert_close`（内部转 fp32）；
- 涉及浮点求和顺序的题目，与 fp64 参考比较而非与 torch fp32 直接比较（见 004 test.py 示范）；
- 采样/随机类题目固定 seed，统计性质单独测（如分布卡方，题 073）。

## 4. 性能达标线（评级）

| 等级 | 标准（相对 torch eager 基线） |
|---|---|
| 合格 | 正确性通过，性能 ≥ 50% |
| 良好 | ≥ 80%，或达到题目 README 给出的绝对带宽/算力线 |
| 优秀 | ≥ torch eager（或 cuBLAS/torch SDPA）的 100% |

每题 README 的「评分标准」一节给出该题的具体达标线；`bench.py` 输出末尾附本次测试硬件与 PyTorch/CUDA 版本，方便横向对比。

## 5. 提交结果

PR 附带 bench 输出（注明 GPU 型号、驱动、CUDA、PyTorch 版本）。格式：

```text
cuda (grid-stride)              1.234 ms |   612.3 GB/s
torch eager                     1.357 ms |   557.5 GB/s
```

欢迎把多卡的对比数据贴进题目 README 的「性能参考」一节（注明环境）。
