# 006 · Softmax（行归一化）

难度：★★ | 章节：第1章 入门 | 标签：`reduction` `safe-softmax` `row-parallel` `fp16`

## 题目

对 `[rows, cols]` 输入的最后一维做 softmax：

$$y_{ij} = \frac{e^{x_{ij} - m_i}}{\sum_{k} e^{x_{ik} - m_i}}, \quad m_i = \max_k x_{ik}$$

```python
def softmax(x: torch.Tensor) -> torch.Tensor  # 形状不变
```

先减行最大值再取指数（**safe softmax**）是必须的数值技巧——不做的话 $e^{88}$ 就溢出 fp32。本题用「一个 block 负责一行」的经典模式，为 031 online-softmax 和 043 FlashAttention 铺路。

## 输入约束

- fp32 与 fp16 两种 dtype 都要支持；
- `x`：二维 `[rows, cols]`，已连续；
- `cols ≤ 16384`（更大的行需要 031 的 online 方案）。

## 要求

**必做：**

1. CUDA 版：**一个 block 负责一行**，两遍扫描（先归约行最大值，再归约 exp 和），复用 004 的 block 归约；
2. **fp16 输入时用 fp32 累加**（读入转 fp32 计算，写出转回 fp16）——这是 LLM 推理 kernel 的标准姿势；
3. Triton 版：一个 program 一行，`BLOCK = next_power_of_2(cols)`；
4. 与 `torch.softmax` 对齐（fp32 与 fp16 各测一组）。

**选做（进阶）：**

1. 不减 max 的"naive softmax"在什么输入下会坏？构造一组数据让测试失败；
2. 实现向量化读写（fp16 用 `half2`）对比带宽；
3. 阅读 [OneFlow 的 softmax 优化文章](https://blog.csdn.net/oneflow_official/article/details/112175731)，解释"批量行 + 向量化 + 打包归约"三级火箭。

## 提示

<details><summary>提示 1：block 大小选择</summary>
binding 里根据 cols 选 block：64 → 1024 的 2 的幂。行短时 block 太小浪费，太大浪费 warp。
</details>

<details><summary>提示 2：共享内存复用</summary>
max 归约和 sum 归约复用同一块 <code>__shared__</code>——第二次写入前记得 <code>__syncthreads()</code>（004 的模板已处理）。
</details>

## 常见陷阱

- 忘减 max：x 大时 `expf` 上溢出 inf，inf/inf = NaN；
- fp16 直接用 `__half` 累加 exp 和：累加误差大且易溢出，必须 fp32 中间量；
- `exp(0)=1` 的被掩码位置混入 sum（Triton 里 `other=-inf` 后 exp 得 0，安全；CUDA 循环以步长跳过超界即可）。

## 评分标准

- 正确性：`python test.py` 全绿（fp32 atol=rtol=1e-5；fp16 atol=rtol=1e-3）；
- 性能：`bytes = 2 × rows × cols × 4`（fp32），fp16 减半；CUDA 版 ≥ torch eager 的 80%。

## 参考资料

- [OneFlow：如何实现一个高效的 Softmax CUDA kernel](https://blog.csdn.net/oneflow_official/article/details/112175731)（中文必读）
- [Triton 教程 02-fused-softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Online normalizer calculation for softmax (arXiv:1805.02867)](https://arxiv.org/abs/1805.02867)（下一题的理论基础）
