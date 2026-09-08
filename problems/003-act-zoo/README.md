# 003 · 激活函数全家桶（Act Zoo）

难度：★ | 章节：第1章 入门 | 标签：`elementwise` `activation` `fusion`

## 题目

实现三个常用激活：**relu / silu / sigmoid**（编号 0/1/2），以及一个 **bias+激活融合**版本：

$$y_{ij} = \text{act}(x_{ij} + b_j)$$

其中 `x` 是 `[rows, cols]`，`b` 是 `[cols]` 广播——这是 LLM MLP 里 `linear → activation` 的真实形态（线性层自带 bias）。

```python
def act(x: torch.Tensor, kind: int) -> torch.Tensor            # 0=relu 1=silu 2=sigmoid
def fused_bias_act(x: torch.Tensor, bias: torch.Tensor, kind: int) -> torch.Tensor
```

## 输入约束

- `x`：fp32 CUDA 张量，`act` 形状任意，`fused_bias_act` 为二维 `[rows, cols]`，已连续；
- `bias`：fp32，形状 `[cols]`。

## 要求

**必做：**

1. CUDA 版：`__device__` 函数 `apply_act(float, int)` 统一三种激活，elementwise kernel + fused kernel 各一个；
2. Triton 版：`kind` 作为 `tl.constexpr`，**编译期**分支（对比 CUDA 的运行期分支，体会两种分发方式的取舍）；
3. 与 `torch.relu / F.silu / F.sigmoid` 对齐。

**选做（进阶）：**

1. 运行期分支每个元素都要判断 `kind`，为什么实测性能几乎无损？（提示：分支对 warp 内所有线程一致）
2. 仿照 vLLM 的做法改成 C++ 模板分发三个 kernel 实例，对比 binary 尺寸与性能；
3. 支持 `x` 非二维任意形状的 `fused_bias_act`（按最后一维广播）。

## 提示

<details><summary>提示 1：silu</summary>
$silu(x) = x \cdot sigmoid(x) = \frac{x}{1+e^{-x}}$。LLaMA 的门控 <code>silu(gate)*up</code>（题 029）就是它的融合应用。
</details>

<details><summary>提示 2：fused 版的列索引</summary>
扁平索引 <code>idx</code> 的列号 <code>col = idx % cols</code>，一行连续存放（row-major）。
</details>

## 常见陷阱

- `expf(-x)` 在 x 为大负数时上溢为 inf → sigmoid 结果为 0，**这是正确行为**，不要"修"它；
- fused 版忘记按 `cols` 取模，bias 索引错乱；
- Triton 里 `tl.constexpr` 的 int 比较 `if kind == 0:` 是编译期展开，运行期传变量会编译错误——这正是教学点。

## 评分标准

- 正确性：`python test.py` 全绿（fp32，atol=rtol=1e-6）；
- 性能：memory-bound，`act: bytes = 2 × n × 4`，`fused: bytes = 2 × rows × cols × 4 + cols × 4`。

## 参考资料

- [vLLM activation kernels](https://github.com/vllm-project/vllm/tree/main/csrc)（`act_and_mul_kernel` 系列的生产级写法）
- [Liger-Kernel 的 silu 实现](https://github.com/linkedin/Liger-Kernel/tree/master/src/liger_kernel/ops)
- PyTorch [Activation Functions 文档](https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions)
