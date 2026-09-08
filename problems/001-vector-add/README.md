# 001 · 向量加法 Vector Add

难度：★ | 章节：第1章 入门 | 标签：`elementwise` `coalesced-access` `grid-stride`

## 题目

给定两个等长 fp32 CUDA 张量 `a`、`b`，计算 `c[i] = a[i] + b[i]`。

这是 GPU 编程的 "Hello World"——但它包含了几乎所有逐元素算子的骨架：**线程到数据的映射、边界保护、以及打满显存带宽**。Softmax、GELU、RMSNorm……大模型里一半的算子都是这道题的变体。

## 函数签名

```python
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor  # 返回新张量
```

## 输入约束

- `a`、`b`：形状相同、已连续的 fp32 CUDA 张量（维度任意，按扁平索引处理）；
- 元素数 `n ∈ [1, 2³¹)`——注意索引类型的选择。

## 要求

**必做：**

1. CUDA kernel 正确处理任意 `n`（尤其不是 block 大小整数倍时）；
2. 实现 **grid-stride loop** 版本（线程数与数据规模解耦）；
3. `python test.py` 全绿。

**选做（进阶）：**

1. 实现 naive（一线程一元素）版本，在大数组上与 grid-stride 对比 `bench.py`，解释差异（或为何几乎无差异）；
2. 支持 fp16 / bf16（用 vectorized `half2` 进一步压榨带宽）；
3. 用 roofline 模型解释：为什么这个算子的理论上限是显存带宽？（见 [docs/benchmark.md](../../docs/benchmark.md)）

## 提示

<details><summary>提示 1：线程与数据映射</summary>
线程 `i` 处理元素 `i`：`i = blockIdx.x * blockDim.x + threadIdx.x`，边界用 `if (i < n)` 保护。
</details>

<details><summary>提示 2：grid-stride loop</summary>
<code>for (long i = start; i < n; i += stride)</code>，其中 <code>stride = gridDim.x * blockDim.x</code>。这样无论启动多少线程都能覆盖全部数据。
</details>

<details><summary>提示 3：内存合并访问</summary>
相邻线程访问相邻地址（thread 0 → a[0]，thread 1 → a[1]…）时硬件自动合并成大事务，这是打满带宽的前提。故意让 thread i 访问 a[i*n] 对比一下带宽。
</details>

## 常见陷阱

- **越界**：`n` 不是 256 的倍数时最后一个块越界写；
- **索引溢出**：用 `int` 当全局索引，`n > 2³¹` 时溢出（用 `long`）；
- **忘记 `.contiguous()`**：调用方传入非连续张量时结果错乱（binding 里应检查或处理）。

## 评分标准

- 正确性：`python test.py` 全绿（fp32，atol=rtol=1e-5）；
- 性能：`python bench.py` 中 CUDA 版带宽 ≥ torch eager 的 90%（RTX 4070 Ti SUPER 参考值：≥ 600 GB/s）。该算子每次调用读 2 个数组、写 1 个数组，`bytes = 3 × n × 4`。

## 性能参考（RTX 4070 Ti SUPER 实测，torch 2.12.0+cu132，N = 16M fp32）

| 实现 | 耗时 | 带宽 |
|---|---|---|
| torch eager | 0.319 ms | 630.5 GB/s |
| cuda（grid-stride） | 0.317 ms | 634.5 GB/s |
| triton | 0.319 ms | 631.8 GB/s |

理论带宽 672 GB/s：好的实现应 ≥ 90%（~600 GB/s）。这条题的带宽就是"天花板"——后面所有 memory-bound 算子都拿它当尺子。

## 参考资料

- NVIDIA 官方样例 [cuda-samples/vectorAdd](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/vectorAdd)
- Triton 官方教程 [01-vector-add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)（[源码](https://github.com/openai/triton/blob/main/python/tutorials/01-vector-add.py)）
- PMPP 第4版 §2.1-2.5（线程组织与映射）
