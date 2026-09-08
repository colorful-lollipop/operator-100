# 007 · RMSNorm

难度：★★ | 章节：第1章 入门 | 标签：`normalization` `row-parallel` `llm-core`

## 题目

实现 RMSNorm（LLaMA / Mistral / Qwen 等几乎所有现代 LLM 的归一化层）：

$$y_i = \frac{x_i}{\sqrt{\frac{1}{H}\sum_{j=1}^{H} x_j^2 + \varepsilon}} \cdot w_i$$

```python
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor
```

相比 LayerNorm（008）它省掉了均值中心化——这正是它流行的原因之一：计算更省、等价性够好。本题还引入 LLM kernel 的另一个关键习惯：**fp16/bf16 输入、fp32 中间计算**。

## 函数签名

- `x`：`[..., H]` fp32 或 fp16 CUDA 张量（本题按二维 `[rows, H]` 处理即可）；
- `w`：`[H]` fp32 权重；
- 输出与 `x` 同 dtype：内部用 fp32 计算，最后转回。

## 要求

**必做：**

1. CUDA 版：一个 block 一行，先归约 $\sum x_j^2$，再写回；fp16 输入走 fp32 累加路径；
2. Triton 版：一行一个 program；
3. 与参考实现对齐（fp32、fp16 各一组）。

**选做（进阶）：**

1. `rsqrtf` vs `1/sqrtf` 的指令级差异；
2. 结合 003 的融合思想，写出 `fused_add_rmsnorm`（残差融合，题 027 的预告）；
3. 阅读 [llm.c 的 layernorm.cuh](https://github.com/karpathy/llm.c/blob/master/llmc/layernorm.cuh)，对比 RMS 版本的简化。

## 提示

<details><summary>提示 1：先归约再写回</summary>
两遍扫描：第一遍 <code>acc += x[i]*x[i]</code>，block 归约得到 sumsq；第二遍 <code>y[i] = x[i] * rsqrtf(sumsq/H + eps) * w[i]</code>。
</details>

<details><summary>提示 2：eps 位置</summary>
eps 在开根号<b>内</b>、除法<b>外</b>：<code>rsqrtf(mean + eps)</code>。写错位置在 x 全零行会 NaN。
</details>

## 常见陷阱

- 在 fp16 半精度里做平方和：H=4096、x~N(0,1) 时平方和 ~4096，fp16 最大值 65504，极易溢出——**必须 fp32 累加**；
- `w` 与输出的 dtype 混算顺序不同引入误差：统一转 fp32 再算，最后一步才 cast；
- 输入全零行：`mean=0`，靠 eps 兜底，验证一下不产生 NaN。

## 评分标准

- 正确性：`python test.py` 全绿（fp32 atol=rtol=1e-5；fp16 atol=rtol=1e-2）；
- 性能：`bytes ≈ 2 × rows × H × 4`（fp32；w 可忽略），CUDA 版 ≥ torch eager 的 85%。

## 性能参考（RTX 4070 Ti SUPER 实测，torch 2.12.0+cu132，8192×4096）

| 实现 | fp32 | fp16 |
|---|---|---|
| torch eager（多 kernel 组合） | 1.501 ms / 179 GB/s | 2.141 ms / 63 GB/s |
| cuda（单 kernel） | 0.428 ms / 628 GB/s | 0.226 ms / 595 GB/s |
| triton（单 kernel） | 0.428 ms / 627 GB/s | 0.214 ms / 627 GB/s |

**fp32 下 3.5×、fp16 下 10× 的差距**来自哪里？eager 的 `pow → mean → rsqrt → mul → mul → cast` 每步都完整读写一次显存，而融合 kernel 只有一次读一次写。这就是"融合"二字的含义，也是 027-030 一整章融合算子的动机。

## 参考资料

- Zhang & Sennrich, [Root Mean Square Layer Normalization (arXiv:1910.07467)](https://arxiv.org/abs/1910.07467)
- [llm.c layernorm.cuh](https://github.com/karpathy/llm.c/blob/master/llmc/layernorm.cuh)
- [Liger-Kernel rms_norm](https://github.com/linkedin/Liger-Kernel/blob/master/src/liger_kernel/ops/rms_norm.py)（Triton 生产实现）
