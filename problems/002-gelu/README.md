# 002 · GELU 激活

难度：★ | 章节：第1章 入门 | 标签：`elementwise` `activation` `math-function`

## 题目

实现 GELU（Gaussian Error Linear Unit）激活函数的两种版本：

- **精确版**（erf）：$\text{gelu}(x) = 0.5x\left(1 + \text{erf}\left(\frac{x}{\sqrt 2}\right)\right)$
- **近似版**（tanh，GPT 系列<BERT/GPT-2/GPT-3>实际使用）：$\text{gelu}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right)$

接口：

```python
def gelu(x: torch.Tensor, mode: int = 0) -> torch.Tensor  # mode: 0=erf 精确版, 1=tanh 近似版
```

## 输入约束

- `x`：fp32 CUDA 张量，形状任意、已连续；
- 两者数值差最大约 0.0003（在 x≈±3 附近），这就是"精度换速度"的直观教材。

## 要求

**必做：**

1. CUDA 版：两个 `__device__` 函数 + 一个 kernel（模板或运行期分支分发），边界保护；
2. Triton 版：一个 kernel，用 `tl.constexpr` 编译期分发两种模式；
3. 分别与 `torch.nn.functional.gelu` 的对应模式对齐（**必须同模式比较**）。

**选做（进阶）：**

1. 比较 erf 版与 tanh 版的 bench 差异；GPT 系为何敢用近似版？
2. 用 fp16 输入再测一遍，观察哪一版先掉精度；
3. 数值讨论：tanh 版在 $|x|$ 很大时 `exp` 会不会溢出？你的写法为什么安全/不安全？

## 提示

<details><summary>提示 1：CUDA 数学函数</summary>
erf 用 <code>erff(x * M_SQRT1_2)</code>（<code>#include &lt;math_constants.h&gt;</code>），tanh 用 <code>tanhf</code>。注意 fp32 版本用 <code>erff</code> 而非 <code>erf</code>。
</details>

<details><summary>提示 2：Triton 里的 tanh</summary>
<code>tl.exp</code>/<code>tl.erf</code> 是基础原语；tanh 可用恒等式 $tanh(v) = 1 - \frac{2}{e^{2v}+1}$ 手写——顺便想想大 $|v|$ 时为什么不会出 NaN。
</details>

## 常见陷阱

- **同模式比较**：拿 erf 实现去对 tanh 模式的 torch 输出，容差再大也对不上；
- Triton 中先 `tl.load(..., other=0.0)` 掩码填充再计算是安全的，但若用 `tl.exp(x)` 且 `other=-inf` 会出 NaN；
- fp16 下 `x*x*x` 在 x≈-3.4e38... 的边缘数值问题（本题 fp32 输入无此忧，选做 fp16 时注意）。

## 评分标准

- 正确性：`python test.py` 全绿（fp32：erf 版 atol=rtol=1e-6；tanh 版 atol=rtol=1e-5）；
- 性能：memory-bound，`bytes = 2 × n × 4`，CUDA 版 ≥ torch eager 的 90%。

## 参考资料

- Triton 教程（旧版 08-gelu-backward 的前向部分已并入官方示例 ⚠️，可读 [04-low-memory-dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html) 的逐元素模式）
- [vLLM activation_kernels](https://github.com/vllm-project/vllm/tree/main/csrc)（生产级实现含 fp16/fp32 dispatch）
- [KernelBench level1/15-18](https://github.com/ScalingIntelligence/KernelBench)（同款算子的评测口径）
- Hendrycks & Gimpel, [Gaussian Error Linear Units (arXiv:1606.08415)](https://arxiv.org/abs/1606.08415)
