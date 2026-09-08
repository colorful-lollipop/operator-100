# 004 · 归约求和 Reduce Sum

难度：★★ | 章节：第1章 入门 | 标签：`reduction` `warp-shuffle` `shared-memory` `multi-pass`

## 题目

对任意规模（最大 2³¹-1 个元素）的 fp32 CUDA 张量求总和，返回 0 维标量张量：

```python
def reduce_sum(x: torch.Tensor) -> torch.Tensor  # 标量 torch.Tensor
```

归约是**所有** block 内聚合型算子（softmax、norm、attention 的行内统计）的地基。Mark Harris 的经典讲义用 7 个版本讲透了它的优化史——本题就是那篇讲义的实战版。

## 输入约束

- `x`：fp32 CUDA 张量，形状任意、已连续，按扁平索引处理；
- 元素数 `n ∈ [0, 2³¹)`。

## 要求

**必做：**

1. CUDA 版实现**两级归约**：第一阶段每个 block 归约一段数据写出部分和，第二阶段单 block 归并所有部分和；
2. block 内使用 **warp shuffle + 共享内存**的树形归约（禁止"单线程串行加全 block"的写法）；
3. 以 fp64 CPU 求和为真值验证（注意：不能拿 torch fp32 的 `sum()` 当真值——它和你的 kernel 只是求和顺序不同）。

**选做（进阶）：**

1. 每线程累加 8 个元素（`items_per_thread=8`）vs 1 个，对比 bench：为什么循环展开后快这么多？
2. 实现 warp 级 grid 归约（单 pass + `atomicAdd`），对比两级方案的确定性与性能；
3. 思考：为什么两次运行结果可能有细微差别？什么时候业务需要**确定性归约**？

## 提示

<details><summary>提示 1：访存模式</summary>
block 内线程按 <code>i = base + k*blockDim + tid</code> 跨步读数据（相邻线程读相邻地址），而不是 <code>i = base + tid*items</code>（相邻线程相隔 items 跳跃）——前者合并访存。
</details>

<details><summary>提示 2：warp shuffle</summary>
<code>__shfl_down_sync(0xffffffff, v, s)</code> 让蝴蝶形归约在 32 线程内寄存器间完成，不碰共享内存；warp 间再用共享内存汇总一次。
</details>

<details><summary>提示 3：共享内存复用</summary>
同一个 <code>__shared__</code> 数组被 max 和 sum 两次归约复用时，第二次写入前必须 <code>__syncthreads()</code>，否则读到上一轮的脏数据（本题 006 会真踩这个坑）。
</details>

## 常见陷阱

- `__shfl_down_sync` 掩码写死 `0xffffffff`：block 不足 32 线程时未定义行为（本题固定 256 线程规避）；
- 两阶段之间忘记 `cudaGetLastError` 检查（第一个 kernel 越界会在第二个 kernel 才炸）；
- fp32 求和误差：n=1M 时不同顺序的结果差可达 ~1e-2 量级——**这是数学属性不是 bug**，容差必须按 docs/benchmark.md 放宽。

## 评分标准

- 正确性：`python test.py` 全绿（与 fp64 真值比：rtol=1e-3, atol=1e-2）；
- 性能：`bytes ≈ n × 4`（只读，输出可忽略），CUDA 版 ≥ torch eager 的 80%（torch 的 reduce 是高度调优的，80% 已是及格线）。

## 性能参考（RTX 4070 Ti SUPER 实测，torch 2.12.0+cu132，N = 64M fp32）

| 实现 | 耗时 | 带宽 |
|---|---|---|
| torch eager | 0.418 ms | 641.8 GB/s |
| cuda（两级归约 + shuffle） | 0.422 ms | 635.6 GB/s |
| triton | 0.419 ms | 641.4 GB/s |

两级归约的手写版达到 torch 的 99%——归约是"优化到头"的经典案例，Harris 讲义里的 7 个版本就是这段历史的浓缩。

## 参考资料

- [Mark Harris《Optimizing Parallel Reduction in CUDA》](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)（必读，7 版本递进）
- [Faster Parallel Reductions on Kepler（NVIDIA Blog）](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)（shuffle 写法出处）
- [gpu-mode/lectures L9: Reductions](https://github.com/gpu-mode/lectures)
