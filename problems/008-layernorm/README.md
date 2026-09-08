# 008 · LayerNorm

难度：★★ | 章节：第1章 入门 | 标签：`normalization` `two-pass` `welford` `llm-core`

## 题目

实现标准 LayerNorm（GPT-2/BERT 时代的标配，训练侧至今主流）：

$$y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot w_i + b_i, \quad \mu = \frac{1}{H}\sum_j x_j,\ \ \sigma^2 = \frac{1}{H}\sum_j (x_j - \mu)^2$$

```python
def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor
```

这是第1章的收官题：综合 004 的归约、006 的行并行、007 的 fp16 处理。**方差用两遍扫描**（先 μ 再 σ²）而非一遍式 $E[x^2]-\mu^2$——后者快但数值不稳，这里的取舍值得细品。

## 函数签名

- `x`：`[rows, H]` fp32 或 fp16 CUDA 张量；`w`、`b`：`[H]` fp32；
- 输出与 `x` 同 dtype。

## 要求

**必做：**

1. CUDA 版：一个 block 一行，**两遍扫描**（先归约 μ，再归约 $\sum(x-\mu)^2$，最后写回），fp16 输入 fp32 累加；
2. Triton 版：一行一个 program；
3. 与 `torch.nn.functional.layer_norm` 对齐。

**选做（进阶）：**

1. 实现一遍式（$E[x^2]-\mu^2$）版本，构造大方差数据（如 x ~ 1e4 量级）让两版误差拉开，解释为什么 PyTorch 内部用 Welford 算法；
2. 实现 Welford 在线方差的 block 归约版；
3. 对比 RMSNorm（007）的 kernel：省掉均值后快了多少？为什么？

## 提示

<details><summary>提示 1：三段式</summary>
第一遍求 μ → 第二遍求 σ² → 第三遍写回。行会从 L2 缓存重复读 2-3 次，所以该算子是"半计算半访存"的。
</details>

<details><summary>提示 2：epilogue</summary>
<code>y[i] = (x[i]-mu) * inv_std * w[i] + b[i]</code>，inv_std = <code>rsqrtf(var + eps)</code>。
</details>

## 常见陷阱

- **一遍式方差误差**：$E[x^2]-\mu^2$ 两个大数相减，x 量级大时灾难性失真（选做题亲手复现）；
- fp16 累加溢出（同 007）；
- `w`/`b` 忘转 fp32 就与 fp16 中间量相乘。

## 评分标准

- 正确性：`python test.py` 全绿（fp32 atol=rtol=1e-4；fp16 atol=rtol=1e-2）；
- 性能：`bytes ≈ 3 × rows × H × 4`（x 读 2 次 + 写 1 次，fp32），CUDA 版 ≥ torch eager 的 70%（LayerNorm 多遍读，torch 内部向量化很强）。

## 参考资料

- [Triton 教程 05-layer-norm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)（含 backward，是题 081 的预告）
- [OneFlow：CUDA 优化之 LayerNorm 性能优化实践](https://blog.csdn.net/oneflow_official/article/details/121974648)
- [NVIDIA/apex fused_layer_norm](https://github.com/NVIDIA/apex)（生产级实现）
- Welford, "Note on a method for calculating corrected sums of squares and products"（在线方差算法）
