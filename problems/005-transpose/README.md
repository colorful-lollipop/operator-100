# 005 · 矩阵转置 Transpose

难度：★★ | 章节：第1章 入门 | 标签：`coalescing` `shared-memory` `bank-conflict`

## 题目

实现行主序 `[rows, cols]` fp32 矩阵的转置：`y[c, r] = x[r, c]`，输出 `[cols, rows]`。

```python
def transpose(x: torch.Tensor) -> torch.Tensor
```

这是**第一个"读合并、写就不合并"的算子**——朴素写法写带宽只有理论值一半，而 shared memory 分块能把两头都变成合并访问。理解本题，就理解了 GPU 内存优化的核心套路。

## 输入约束

- `x`：fp32 CUDA 张量，二维 `[rows, cols]`，已连续；
- `rows, cols ∈ [1, 65536]`。

## 要求

**必做：**

1. CUDA 版实现两个 kernel：
   - `naive`：直接按转置下标读写；
   - `tiled`：32×32 分块经过共享内存中转，**并用 `[TILE][TILE+1]` padding 消除 bank conflict**；
2. 两个版本都要通过测试，bench 里对比两者带宽；
3. Triton 版：二维网格分块 + `tl.trans`。

**选做（进阶）：**

1. 用 `nsys` / `ncu` 观察 naive 版的 uncoalesced 访存事务数；
2. 实验 padding 换成 `+2`、`+4` 的效果；
3. 对角化 block 分配（NVIDIA 博客里的进阶方案）能否进一步降低 bank conflict？

## 提示

<details><summary>提示 1：为什么 naive 慢</summary>
读 <code>x[r*cols+c]</code>：相邻线程 c 相邻 → 合并；写 <code>y[c*rows+r]</code>：相邻线程写步长 rows 的地址 → 分散成 32 个独立事务。
</details>

<details><summary>提示 2：shared memory 中转</summary>
按合并方式读入 tile，<code>__syncthreads()</code>，再按"转置后的合并方式"写出：<code>y[(c0+ty)*rows + r0+tx] = tile[tx][ty]</code>。
</details>

<details><summary>提示 3：+1 padding 的魔法</summary>
<code>tile[32][32]</code> 时 warp 读 <code>tile[tx][ty]</code> 列方向正好落同一 bank（32 列 = 32 bank）；改成 <code>tile[32][33]</code> 后错位，冲突消除。
</details>

## 常见陷阱

- tiled 版读写两侧的**边界掩码都要做**（rows/cols 非整除 32）；
- `__syncthreads()` 漏写或位置错；
- 输出张量形状忘转置（还是 `[rows, cols]`）；
- 行/列索引 `(long)` 溢出：rows×cols > 2³² 时 `r*cols` 用 int 算会溢出。

## 评分标准

- 正确性：`python test.py` 全绿（fp32，atol=rtol=0——转置是精确操作，容差为 0 才对）；
- 性能：`bytes = 2 × rows × cols × 4`，**tiled 版带宽 ≥ naive 版的 2 倍**（RTX 4070 Ti SUPER 上 naive 约 300 GB/s，tiled 应接近 600 GB/s）。

## 参考资料

- [An Efficient Matrix Transpose in CUDA C++（NVIDIA Blog）](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- PMPP 第4版 §4.4（tiled 转置）
- [cuda-samples matrixTranspose](https://github.com/NVIDIA/cuda-samples)（六个版本的官方演进）
