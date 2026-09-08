# 算子 100 题总览（CURRICULUM）

> 难度：★ 入门 · ★★ 基础 · ★★★ 进阶 · ★★★★ 困难 · ★★★★★ 专家
> 状态：✅ 已实现（题面 + 参考实现 + CUDA + Triton + 测试 + 基准） · 🔲 待实现（欢迎认领）
> 每题约定：PyTorch 参考实现为正确性基准；CUDA / Triton 为必写方案；第11章部分题目为指定后端（昇腾等）。

## 第1章 · 入门：元素级算子与基础模式（001-008）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 001 | [vector-add](problems/001-vector-add/README.md) 向量加法 | ★ | `c[i]=a[i]+b[i]`，任意长度 | 线程映射、边界保护、grid-stride loop、内存合并访问 | [cuda-samples]、[triton-tut1] | ✅ |
| 002 | [gelu](problems/002-gelu/README.md) GELU 激活 | ★ | 实现 erf 精确版与 tanh 近似版 GELU | 逐元素数学函数、精度-性能取舍、`erff/tanhf` | [triton-tut]、[vLLM]、[KernelBench] | ✅ |
| 003 | [act-zoo](problems/003-act-zoo/README.md) 激活全家桶 | ★ | relu/silu/sigmoid + `act(x+bias)` 融合 | 编译期 vs 运行期分发、broadcast 加 bias | [vLLM activation]、[Liger] | ✅ |
| 004 | [reduce-sum](problems/004-reduce-sum/README.md) 归约求和 | ★★ | 全 tensor 求和（任意规模） | 两级归约、warp shuffle、共享内存树形归约 | [reduction]、[kepler-reduce]、[gpu-mode L9] | ✅ |
| 005 | [transpose](problems/005-transpose/README.md) 矩阵转置 | ★★ | 转置大矩阵，读合并写也合并 | 合并访存、shared memory 分块、bank conflict 消除 | [transpose-blog]、[PMPP] | ✅ |
| 006 | [softmax](problems/006-softmax/README.md) 行 Softmax | ★★ | 每行 safe softmax | 块级归约、safe softmax（减最大值）、fp16 累加 fp32 | [OneFlow-softmax]、[triton-tut2] | ✅ |
| 007 | [rmsnorm](problems/007-rmsnorm/README.md) RMSNorm | ★★ | LLaMA 系标配：`x·w/rsqrt(mean(x²)+ε)` | 行归约、eps 细节、fp16 计算 fp32 | [llm.c]、[Liger]、[vLLM] | ✅ |
| 008 | [layernorm](problems/008-layernorm/README.md) LayerNorm | ★★ | `(x-μ)/√(σ²+ε)·w+b` | 两遍均值/方差、与 RMSNorm 的性能差异 | [triton-tut5]、[OneFlow-ln]、[apex] | ✅ |

## 第2章 · 内存与索引：gather/scatter/scan（009-016）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 009 | embedding-lookup | ★★ | 按 token id 查词向量表 | gather、索引越界检查、padding_idx | [llm.c encoder]、[KernelBench] | 🔲 |
| 010 | scatter-add | ★★ | `out[idx] += src`（index_add） | 原子加、写冲突、确定性讨论 | [torch scatter]、[KernelBench] | 🔲 |
| 011 | histogram | ★★ | 统计 token id 直方图（词表 128K） | atomicAdd、shared memory 直方图、全局归并 | [PMPP ch9]、[cuda-samples] | 🔲 |
| 012 | causal-mask | ★ | 生成下三角因果掩码（0/-inf） | 二维网格映射、分支消除 | [llm.c]、[vLLM] | 🔲 |
| 013 | prefix-sum | ★★★ | 一维前缀和（含/不含自身） | Hillis-Steele vs Blelloch 扫描、跨块扩展 | [gpugems3-39] | 🔲 |
| 014 | cumsum-2d | ★★★ | 沿最后一维的 2D 分段扫描 | 分段扫描、变长序列应用 | [gpugems3-39]、[FA] | 🔲 |
| 015 | topk | ★★★ | Top-K（k≤32），返回值与索引 | bitonic 排序、阈值筛选、稳定性 | [vLLM topk]、[KernelBench] | 🔲 |
| 016 | varlen-pack | ★★ | 由 seq_lens 构建 cu_seqlens 并打包 token | 变长布局、前缀和辅助、FlashAttention varlen | [FA varlen]、[vLLM] | 🔲 |

## 第3章 · GEMM：从朴素到 Tensor Core（017-026）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 017 | sgemm-naive | ★★ | 一线程一输出的 fp32 GEMM | 内存访问模式分析、为何慢 | [siboehm]、[PMPP ch6] | 🔲 |
| 018 | sgemm-tiled | ★★★ | shared memory 分块 GEMM | tile 缓存、合并读、全局内存访问减半 | [siboehm]、[gpu-mode L3] | 🔲 |
| 019 | sgemm-register | ★★★★ | 一线程算 TM×TN 微分块 | 寄存器分块、算术强度、bank conflict 调优 | [siboehm]（重点对照） | 🔲 |
| 020 | sgemm-vectorized | ★★★ | float4 向量化访存 + 边界处理 | 向量化 load/store、对齐、尾部处理 | [siboehm]、[cutlass 00] | 🔲 |
| 021 | sgemm-double-buffer | ★★★★ | cp.async / 手工双缓冲流水线 | 异步拷贝、计算与访存重叠 | [cutlass]、[PMPP] | 🔲 |
| 022 | hgemm-tensorcore | ★★★★ | bf16 WMMA/MMA Tensor Core GEMM | wmma API、fragment、mixed precision | [cutlass 07/14]、[NVIDIA wmma] | 🔲 |
| 023 | sgemm-splitk | ★★★ | 瘦矩阵（M 小 K 大）GEMM | split-K、原子加归并、occupancy | [cutlass]、[KernelBench] | 🔲 |
| 024 | bmm-strided | ★★ | strided batched GEMM（注意力里的 QKᵀ） | 批量布局、stride 语义 | [llm.c matmul]、[cublas docs] | 🔲 |
| 025 | gemv | ★★ | decode 阶段 GEMV（行内归约） | 行并行、读写带宽比、权重常驻 | [llama.cpp]、[vLLM] | 🔲 |
| 026 | gemm-epilogue | ★★★★ | GEMM + bias + GELU epilogue 融合 | epilogue 融合、中间张量省略 | [cutlass 113]、[triton-tut3] | 🔲 |

## 第4章 · LLM 基础融合算子（027-036）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 027 | fused-add-rmsnorm | ★★★ | `rmsnorm(x + residual)` 并更新 residual | 残差复用、单 kernel 融合、读写省一半 | [Liger]、[vLLM] | 🔲 |
| 028 | fused-add-layernorm | ★★★ | 同上，LayerNorm 版 | 融合模式归纳 | [apex]、[vLLM] | 🔲 |
| 029 | swiglu | ★★ | `silu(gate) * up` 门控激活（两输入融合） | 双 tensor 逐元素融合、LLaMA MLP 核心 | [Liger swiglu]、[vLLM] | 🔲 |
| 030 | fused-bias-act | ★★ | bias + 激活（多激活统一入口） | 融合粒度、kernel 数量与 launch 开销 | [vLLM]、[FlagGems] | 🔲 |
| 031 | online-softmax | ★★★ | 流式（分块）safe softmax，支持超长行 | online max/sum 重缩放、为 FA 铺路 | [online-softmax]（论文）、[OneFlow-softmax] | 🔲 |
| 032 | softmax-causal | ★★ | 因果掩码 + softmax 单 pass | 掩码跳过计算、行内截断 | [triton-tut2]、[llm.c] | 🔲 |
| 033 | log-softmax | ★★ | log_softmax 及数值细节 | log-sum-exp、稳定性 | [torch docs]、[KernelBench] | 🔲 |
| 034 | cross-entropy | ★★★ | 交叉熵前向（vocab 128K 大行） | 大行归约、target 索引、数值稳定 | [Liger CE]、[llm.c fused_classifier] | 🔲 |
| 035 | fused-linear-ce | ★★★★ | lm_head GEMM + CE 分块融合省显存 | chunked 计算避免 logits 全量落地 | [Liger FLCE] | 🔲 |
| 036 | dropout | ★★★ | 融合 dropout（philox RNG 可复现） | GPU 随机数、philox、train/eval 一致性 | [triton-tut dropout]、[torch philox] | 🔲 |

## 第5章 · RoPE 与位置编码（037-040）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 037 | rope | ★★ | GPT-NeoX 风格 RoPE（半旋转） | 复数旋转、cos/sin 缓存、向量化 | [llm.c]、[Liger rope]、[vLLM pos-enc] | 🔲 |
| 038 | rope-interleaved | ★★ | GPT-J 风格（相邻交错）与 037 互转 | 两种布局兼容、内存访问模式对比 | [HF rope 文档]、[vLLM] | 🔲 |
| 039 | rope-qkv | ★★★ | QKV 投影后布局 + position_ids 融合应用 | 多头/多 KV 头（GQA）索引、变长位置 | [vLLM]、[Liger] | 🔲 |
| 040 | alibi-bias | ★★ | ALiBi 加性偏置 fused 到 score/softmax | 斜率计算、加性偏置通用化 | [ALiBi 论文 2108.12409]、[vLLM] | 🔲 |

## 第6章 · Attention：从朴素到 FlashAttention 与 PagedAttention（041-052）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 041 | attn-naive | ★★★ | 朴素注意力前向（物化 S/P 矩阵） | QKᵀ+softmax+V 三步、显存瓶颈测量 | [llm.c attention]、[FA-min] | 🔲 |
| 042 | attn-causal | ★★★ | 因果掩码注意力前向 | 掩码下三角利用、块内跳过 | [llm.c]、[triton-tut6] | 🔲 |
| 043 | flash-attn-v1 | ★★★★ | FlashAttention 前向（tiling + online softmax） | online softmax、rescale、HBM 读写 O(N) | [FA 论文]、[FA-min]、[FA-tiny] | 🔲 |
| 044 | flash-attn-v2 | ★★★★ | FA-2 前向（按行并行、减 rescale） | 线程块划分、softmax 统计量下沉 | [FA2 论文]、[FA-zh] | 🔲 |
| 045 | flash-attn-gqa | ★★★★ | GQA/MQA 支持（多 Q 头共享 KV） | 头映射、decode 场景并行策略 | [GQA 论文 2305.13245]、[FA2 §GQA] | 🔲 |
| 046 | flash-attn-bwd | ★★★★★ | FlashAttention 反向 | 重计算、dQ/dK/dV 三 kernel、原子/分离归约 | [FA 论文 §bwd]、[triton-tut6] | 🔲 |
| 047 | kv-cache-append | ★★ | 把新 KV 原地写入 cache（自回归） | cache 布局、读写竞争、变长批量 | [vLLM cache]、[llama.c?] | 🔲 |
| 048 | paged-attn-v1 | ★★★★ | 分页 KV cache 注意力（block table 间接寻址） | 分页寻址、gather KV | [PagedAttn 论文]、[vLLM kernel]、[PagedAttn-zh] | 🔲 |
| 049 | paged-attn-v2 | ★★★★ | v2：序列维 split-K 并行 | split-K over seq、logits 归并 | [PagedAttn 论文 §v2]、[vLLM] | 🔲 |
| 050 | attn-sliding-window | ★★★ | 滑窗注意力前向（window=4096） | 窗口掩码、KV 读范围裁剪 | [Mistral 论文]、[FA window] | 🔲 |
| 051 | attn-alibi-gqa | ★★★ | ALiBi + GQA 组合注意力 | 偏置与头映射组合 | [vLLM] | 🔲 |
| 052 | attn-tree | ★★★★★ | 投机解码树注意力（tree mask 验证） | 树掩码、批量验证 | [SpecInfer 2310.09761]、[Medusa] | 🔲 |

## 第7章 · 量化与低精度（053-060）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 053 | quant-int8 | ★★ | per-tensor / per-channel 对称量化与反量化 | scale 计算、取整与饱和 | [SmoothQuant]、[torch quant] | 🔲 |
| 054 | quant-group | ★★★ | 分组量化（group=128，scale+zero） | groupwise 误差、布局重排 | [GPTQ]、[AWQ] | 🔲 |
| 055 | gemm-int8 | ★★★★ | W8A8 INT8 GEMM（dp4a/IMMA） | 整数矩阵乘、dp4a intrinsics、反量化 epilogue | [SmoothQuant]、[vLLM cutlass] | 🔲 |
| 056 | gemv-w4a16 | ★★★★ | weight-only INT4 GEMV（groupwise 反量化） | 半字节解包、反量化在 load 路径、AWQ/GPTQ 思想 | [AWQ 论文]、[GPTQ 论文] | 🔲 |
| 057 | gemm-smoothquant | ★★★★ | W8A8 + 融合 per-channel 反量化 | SmoothQuant 尺度迁移思想 | [SmoothQuant 论文] | 🔲 |
| 058 | gemm-fp8 | ★★★★ | FP8(e4m3) per-tensor scale GEMM | FP8 格式、Hopper FP8 路径、scale 处理 | [TE docs]、[DeepSeek-V3 §FP8] | 🔲 |
| 059 | kv-cache-quant | ★★★ | KV cache INT8/FP8 量化与反量化 kernel | cache 精度 trade-off、per-head scale | [vLLM kv quant]、[KIVI 论文] | 🔲 |
| 060 | gemm-w4a16-gptq | ★★★★★ | GPTQ 风格 W4A16（g_idx 重排 + 零点） | g_idx、零点处理、repack 思想 | [GPTQ 论文]、[AutoGPTQ] | 🔲 |

## 第8章 · MoE（061-068）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 061 | moe-topk-softmax | ★★ | 路由：softmax + topk 融合 | 行内 topk、renormalize | [vLLM moe]、[Mixtral 论文] | 🔲 |
| 062 | moe-align-block-size | ★★★ | 按 expert 统计并对齐到 block 边界 | 计数排序思想、对齐布局 | [vLLM moe_align]、[SGLang] | 🔲 |
| 063 | moe-permute | ★★★ | token 按 expert 重排（gather 带映射表） | 索引重排、反向映射保存 | [vLLM]、[DeepSpeed-MoE] | 🔲 |
| 064 | moe-grouped-gemm | ★★★★ | 分 expert 的 grouped GEMM | 变长分组 GEMM、对齐与 padding | [vLLM fused_moe]、[Megablocks] | 🔲 |
| 065 | moe-unpermute | ★★★ | unpermute + scatter add 回原位置 | 原子 vs 排序确定性 | [vLLM] | 🔲 |
| 066 | moe-capacity | ★★★ | capacity/dropless 掩码生成 | 专家容量、负载不均处理 | [Switch Transformer]、[DeepSpeed-MoE] | 🔲 |
| 067 | moe-shared-expert | ★★ | shared expert 输出与 routed 输出融合加法 | 双路径融合 epilogue | [DeepSeek-V2 §shared] | 🔲 |
| 068 | moe-noaux-routing | ★★★ | DeepSeek 风格 sigmoid+bias 路由 | 无 aux loss 路由、bias 更新外的 kernel 侧 | [DeepSeek-V3 §MoE] | 🔲 |

## 第9章 · 解码与采样（069-078）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 069 | argmax | ★ | 全 tensor argmax（带 tie-break） | 块归约携带索引 | [vLLM sampling] | 🔲 |
| 070 | logits-process | ★★ | temperature/重复惩罚/logit bias 融合 | 采样前处理融合、原地操作 | [vLLM logits processors]、[HF] | 🔲 |
| 071 | topk-sampling | ★★★ | top-k 采样端到端 | topk + softmax + 多项式组合 | [vLLM] | 🔲 |
| 072 | topp-sampling | ★★★★ | nucleus（top-p）采样 | 排序/阈值、双 kernel 方案 | [vLLM] | 🔲 |
| 073 | multinomial | ★★★ | 多项式采样（前缀和 + 二分） | 并行随机数、二分查找 | [torch multinomial]、[curand] | 🔲 |
| 074 | sampling-fused | ★★★★ | temperature+top-k+top-p 单 kernel 链 | 采样流水线融合、kernel launch 优化 | [vLLM V1 sampler] | 🔲 |
| 075 | kv-block-mgmt | ★★ | 分页 KV 的 block 分配/释放/映射表维护 | 显存管理逻辑 kernel 化 | [PagedAttn 论文 §3] | 🔲 |
| 076 | spec-verify | ★★★ | 投机解码批量验证（draft vs target 比较） | 接受/拒绝判定、贪心+采样两模式 | [SpecInfer]、[Medusa] | 🔲 |
| 077 | entropy-ppl | ★★ | 熵/困惑度统计 kernel | 大行归约、log 累加 | [FlagGems] | 🔲 |
| 078 | flash-decoding | ★★★★ | 长上下文单 query 解码注意力（split-K over seq） | 跨块 online softmax 归并 | [FA-decoding blog]、[vLLM] | 🔲 |

## 第10章 · 训练侧算子：backward 与优化器（079-086）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 079 | gelu-backward | ★★ | GELU 反向 | 导数推导、tanh 近似的导数 | [triton-tut8 历史]、[Liger] | 🔲 |
| 080 | softmax-backward | ★★ | softmax 反向：`(g-(g·y)y)·y`? 逐行标量技巧 | Jacobian-vector product 化简 | [torch autograd] | 🔲 |
| 081 | layernorm-backward | ★★★★ | LayerNorm 反向（dX 与 dW/dB 分离） | Welford/两遍、dW 跨行归约（原子/分桶） | [triton-tut5 bwd]、[apex] | 🔲 |
| 082 | rmsnorm-backward | ★★★ | RMSNorm 反向 | 与 LN 反向对比简化 | [Liger] | 🔲 |
| 083 | rope-backward | ★★★ | RoPE 反向（旋转是正交变换） | 正交性利用、复用前向 cos/sin | [Liger] | 🔲 |
| 084 | embedding-backward | ★★★ | 词表梯度（scatter add vs 排序确定性） | 原子竞争、deterministic 模式 | [torch embedding bwd] | 🔲 |
| 085 | adamw-fused | ★★★ | 融合 AdamW（多张量 foreach） | 逐元素优化器、多 tensor 打包 launch | [llm.c adamw]、[apex]、[gpu-mode L6] | 🔲 |
| 086 | grad-clip | ★★★ | 全局梯度范数 + 缩放（两阶段） | 范数归约、读写两遍 | [llm.c global_norm] | 🔲 |

## 第11章 · 系统、多后端与毕业设计（087-100）

| # | 题目 | 难度 | 一句话题面 | 核心知识点 | 出处 | 状态 |
|---|---|---|---|---|---|---|
| 087 | stream-overlap | ★★ | 双 stream 重叠 H2D 拷贝与计算 | stream/event、伪依赖分析 | [cuda guide streams] | 🔲 |
| 088 | cuda-graphs | ★★★ | 用 CUDA Graph 捕获 decode step 消 launch 开销 | Graph capture、静态形状约束 | [cuda guide graphs]、[vLLM] | 🔲 |
| 089 | persistent-mlp | ★★★★ | 持久化 kernel：整层 MLP 融合（GEMM+act+GEMM） | persistent CTAs、片上中间结果 | [triton-tut9]、[FA3 思想] | 🔲 |
| 090 | triton-autotune | ★★ | Triton autotune vs 手工 tile 对比 GEMM | 自动调优、配置空间、缓存 | [triton-tut3]、[FlagGems] | 🔲 |
| 091 | inductor-repro | ★★★ | 阅读 torch.compile 生成的 kernel 并手写复现 | inductor 代码生成、对照学习 | [torch tutorial] | 🔲 |
| 092 | custom-op | ★★ | 把手写 kernel 注册为 torch 自定义算子（含 autograd） | TORCH_LIBRARY、opcheck | [torch custom op] | 🔲 |
| 093 | hip-port | ★★★ | 把 005/017 移植到 HIP/ROCm（海光 DCU 同源） | hipify、CUDA/HIP 差异 | [ROCm docs] | 🔲 |
| 094 | ascendc-vector-add | ★★★ | 昇腾 Ascend C：向量加法 | Ascend C 编程模型、Global/Tensor 概念 | [AscendC 指南]、[cann-samples] | 🔲 |
| 095 | ascendc-rmsnorm | ★★★ | 昇腾 Ascend C：RMSNorm | Reduce/标量接口、行归约 NPU 实现 | [AscendC 指南]、[cann-ops] | 🔲 |
| 096 | ascendc-flash-attn | ★★★★★ | 昇腾 Ascend C：FlashAttention 精简版 | Cube/Vector 分工、Tiling 结构 | [cann-samples]、[triton-ascend] | 🔲 |
| 097 | triton-ascend-softmax | ★★★ | Triton-Ascend 后端：同一份 Triton softmax 跑 NPU | 后端可移植性、算子差异 | [triton-ascend] | 🔲 |
| 098 | mini-decode-engine | ★★★★★ | 毕业设计：迷你解码引擎（paged KV + 融合算子 + 采样闭环） | 端到端整合、静态调度 | [vLLM 架构]、[PagedAttn] | 🔲 |
| 099 | transformer-layer | ★★★★★ | 毕业设计：手写整层 Transformer 前向（全融合 kernel） | 算子编排、显存规划 | [llm.c]、[NanoGPT speedrun] | 🔲 |
| 100 | fa3-style | ★★★★★ | 毕业设计：FA-3 风格流水线（warp specialization/ping-pong 思想） | producer-consumer、异步流水线 | [FA3 论文 2407.08608]、[cutlass 113] | 🔲 |

## 引用出处一览

**论文**：[FA] [FlashAttention, arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · [FA2] [FlashAttention-2, arXiv:2307.08691](https://arxiv.org/abs/2307.08691) · [FA3] [FlashAttention-3, arXiv:2407.08608](https://arxiv.org/abs/2407.08608) · [PagedAttn] [vLLM/PagedAttention, arXiv:2309.06180](https://arxiv.org/abs/2309.06180) · [online-softmax] [Online normalizer calculation for softmax, arXiv:1805.02867](https://arxiv.org/abs/1805.02867) · [AWQ] [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) · [GPTQ] [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) · [SmoothQuant] [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) · [ALiBi] [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) · [GQA] [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) · [SpecInfer] [arXiv:2310.09761](https://arxiv.org/abs/2310.09761) · [KIVI] [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)

**开源仓库**：[cuda-samples] [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) · [triton-tut] [openai/triton tutorials](https://github.com/openai/triton/tree/main/python/tutorials) · [KernelBench] [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) · [FlagGems] [FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems) · [Liger] [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) · [FA-min] [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) · [FA-tiny] [66RING/tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) · [llm.c] [karpathy/llm.c](https://github.com/karpathy/llm.c) · [vLLM] [vllm-project/vllm csrc](https://github.com/vllm-project/vllm/tree/main/csrc) · [cutlass] [NVIDIA/cutlass examples](https://github.com/NVIDIA/cutlass/tree/main/examples) · [gpu-mode] [gpu-mode/lectures](https://github.com/gpu-mode/lectures) · [apex] [NVIDIA/apex](https://github.com/NVIDIA/apex) · [cann-samples] [GitCode CANN samples](https://gitcode.com/cann) · [cann-ops] [gitee ascend/cann-ops](https://gitee.com/ascend/cann-ops) · [triton-ascend] [Ascend triton-ascend](https://gitcode.com/Ascend/triton-ascend)

**博客/课程/教材**：[siboehm] [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) · [reduction] [Optimizing Parallel Reduction in CUDA (Mark Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) · [kepler-reduce] [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) · [transpose-blog] [An Efficient Matrix Transpose in CUDA](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) · [gpugems3-39] [GPU Gems 3 Ch.39 Parallel Prefix Sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) · [PMPP] Hwu/Kirk/El Hajj《Programming Massively Parallel Processors》4th ed., 2022 · [OneFlow-softmax] [OneFlow：如何实现一个高效的 Softmax CUDA kernel](https://blog.csdn.net/oneflow_official/article/details/112175731) · [OneFlow-ln] [OneFlow：CUDA 优化之 LayerNorm](https://blog.csdn.net/oneflow_official/article/details/121974648) · [FA-zh] [DefTruth：从 Online-Softmax 到 FlashAttention V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063) · [PagedAttn-zh] [vLLM 核心技术 PagedAttention 原理详解](https://blog.csdn.net/cr7258/article/details/148266171) · [AISystem] [Infrasys-AI/AISystem](https://github.com/chenzomi12/AISystem) · [AscendC 指南] [Ascend C 算子开发指南](https://www.hiascend.com/document/detail/zh/CANNCommercial/810/opdevg/ascendcopdevg/atlas_ascendc_10_0001.html)

> 个别引用项（如 cuda-samples 子目录、cann-ops）随上游演进可能变动路径，以仓库当前结构为准；完整注释版资源地图见 [docs/references.md](docs/references.md)。
