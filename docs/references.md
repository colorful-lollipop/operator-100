# 参考资源地图（References）

> 全部链接于 2026-09 经人工核实可访问；标注 ⚠️ 的条目随上游演进可能变动。
> 用途：为每道题挑选「读什么」——论文讲原理、仓库讲工程、博客讲推导过程。

## 1. 教材与系统课程

| 资源 | 说明 |
|---|---|
| [PMPP 第4版](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) | Hwu, Kirk, El Hajj, *Programming Massively Parallel Processors*, 4th ed., Morgan Kaufmann, 2022（ISBN 978-0-323-91231-0）。GPU 编程标准教材，本项目第1-3章的底层逻辑 |
| [gpu-mode/lectures](https://github.com/gpu-mode/lectures)（原 cuda-mode，已更名，旧链接自动重定向） | 社区 GPU 课程 L1-L28：L1 性能分析与 PyTorch 集成、L3 CUDA 入门、L4 计算与内存架构、L6 优化器、L7 量化、L9 Reductions、L10 生产级库…… Apache-2.0 |
| [Infrasys-AI/AISystem](https://github.com/chenzomi12/AISystem)（原 chenzomi12/AISystem） | 中文《AI系统》开源课程（17k+★）：AI 芯片、编译器、推理引擎与算子全栈，Apache-2.0 |
| [GPU MODE 中文视频（B站）](https://www.bilibili.com/video/BV1QZ421N7pT/) | CUDA/GPU 编程 1-53 课中英字幕 |
| [CUDA MODE 中文配音（B站）](https://www.bilibili.com/video/BV1cgbL6XEfa/) | 42 讲，含 Triton/CUTLASS |

## 2. 官方样例与算子库（出题素材）

| 资源 | 说明 |
|---|---|
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | 官方样例；入门见 `cpp/0_Introduction/vectorAdd`、`cpp/0_Introduction/matrixTranspose` |
| [openai/triton tutorials](https://github.com/openai/triton/tree/main/python/tutorials) | Triton 官方教程（在线版：[triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/index.html)）：01-vector-add、02-fused-softmax、03-matrix-multiplication、04-low-memory-dropout、05-layer-norm、06-fused-attention、07-extern-functions、08-grouped-gemm、09-persistent-matmul、10-block-scaled-matmul、11-programmatic-dependent-launch。注意：旧版 07-dropout/08-gelu-backward 已移除或改号 ⚠️ |
| [Triton 官方文档中文镜像](https://triton.hyper.ai/)（源码 [hyperai/triton-cn](https://github.com/hyperai/triton-cn)，MIT） | 中文阅读友好 |
| [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)（Stanford，MIT） | "Can LLMs Write GPU Kernels?"：level1=100 道单算子、level2=100 道融合组合、level3=50 道整模型。本题库第4-10章的重要出题参考 |
| [FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems)（Apache-2.0，中文 README 见 [Gitee 镜像](https://gitee.com/flagopen/FlagGems)） | BAAI 出品的 Triton 大模型算子库，经 PyTorch ATen 后端零改动替换，700+ 算子模块 |
| [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)（BSD-2） | LLM 训练 Triton 算子：rms_norm、rope、swiglu/geglu、cross_entropy、fused_linear_cross_entropy、fused_moe 等——第4/5/8/10章的直接参照 |
| [NVIDIA/cutlass examples](https://github.com/NVIDIA/cutlass/tree/main/examples)（BSD-3） | 精读建议：00_basic_gemm → 04_tile_iterator → 07_volta_tensorop_gemm → 14_ampere_tf32 → 113_hopper_gemm_activation_fusion |

## 3. 大模型引擎源码（进阶读物）

| 资源 | 说明 |
|---|---|
| [karpathy/llm.c](https://github.com/karpathy/llm.c)（MIT） | 手写 GPT-2 训练：`llmc/` 下 matmul.cuh、attention.cuh、gelu.cuh、layernorm.cuh、adamw.cuh、encoder.cuh、fused_classifier.cuh、global_norm.cuh——最干净的端到端手写 CUDA 教材 |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)（BSD-3） | FA2 为主线；FA3 beta 在 `hopper/`；FA4 面向 Hopper/Blackwell |
| [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) | ~100 行 CUDA 的极简 FA 前向（1.2k★） |
| [66RING/tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) | python/triton/cuda/cutlass 四种实现的 FA 教程（与上一条易混淆） |
| [vllm-project/vllm csrc](https://github.com/vllm-project/vllm/tree/main/csrc)（Apache-2.0） | 生产级 kernel：`attention/`（PagedAttention 在 v0.10.2 为 `attention/attention_kernels.cuh` ⚠️）、`moe/`、`quantization/`、activation/layernorm/pos_encoding kernels（目录随版本重组 ⚠️） |

## 4. 经典文章 / 讲义（逐题精读）

| 资源 | 对应题目 |
|---|---|
| [Mark Harris《Optimizing Parallel Reduction in CUDA》](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) | 004 归约：7 个递进优化版本 |
| [Faster Parallel Reductions on Kepler（NVIDIA Blog）](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) | 004：warp shuffle 归约 |
| [An Efficient Matrix Transpose in CUDA（NVIDIA Blog）](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) | 005：shared memory 转置 + padding 消 bank conflict |
| [Simon Boehm《How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog》](https://siboehm.com/articles/22/CUDA-MMM) | 017-021：GEMM 优化全流程（1 TFLOP→~80% cuBLAS） |
| [GPU Gems 3 Ch.39《Parallel Prefix Sum (Scan) with CUDA》](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) | 013/014：Hillis-Steele vs Blelloch |
| [Milakov & Gimelshein《Online normalizer calculation for softmax》(arXiv:1805.02867)](https://arxiv.org/abs/1805.02867) | 031/043：online softmax 原始论文 |

## 5. 核心论文（按章）

- **Attention**：[FlashAttention arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · [FA-2 arXiv:2307.08691](https://arxiv.org/abs/2307.08691) · [FA-3 arXiv:2407.08608](https://arxiv.org/abs/2407.08608) · [GQA arXiv:2305.13245](https://arxiv.org/abs/2305.13245) · [vLLM/PagedAttention arXiv:2309.06180](https://arxiv.org/abs/2309.06180) · [SpecInfer arXiv:2310.09761](https://arxiv.org/abs/2310.09761) · [ALiBi arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- **量化**：[GPTQ arXiv:2210.17323](https://arxiv.org/abs/2210.17323) · [AWQ arXiv:2306.00978](https://arxiv.org/abs/2306.00978) · [SmoothQuant arXiv:2211.10438](https://arxiv.org/abs/2211.10438) · [KIVI（KV cache 量化）arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **MoE**：[Switch Transformer](https://arxiv.org/abs/2101.03961) · [Mixtral](https://arxiv.org/abs/2401.04088) · [Megablocks](https://arxiv.org/abs/2211.15841) · [DeepSeek-V3 技术报告（FP8/MoE 路由）](https://arxiv.org/abs/2412.19437)

## 6. 中文社区精选

| 资源 | 说明 |
|---|---|
| [OneFlow：如何实现一个高效的 Softmax CUDA kernel](https://blog.csdn.net/oneflow_official/article/details/112175731) | 006/031 的中文精读首选（CC BY-SA 4.0） |
| [OneFlow：CUDA 优化之 LayerNorm 性能优化实践](https://blog.csdn.net/oneflow_official/article/details/121974648) | 008 的向量化和行归约技巧 |
| [DefTruth：Attention 优化——从 Online-Softmax 到 FlashAttention V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063) | 043-046 中文推导系列（知乎，转载见 CSDN） |
| [vLLM 核心技术 PagedAttention 原理详解](https://blog.csdn.net/cr7258/article/details/148266171) | 048/049 中文原理讲解 |

## 7. PyTorch / Triton 官方文档

- [Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) —— 本项目 `solutions/cuda/solution.py` 的加载方式
- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Custom Operators（TORCH_LIBRARY / opcheck）](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) —— 题 092
- [Triton tutorials 索引](https://triton-lang.org/main/getting-started/tutorials/index.html) / [安装](https://triton-lang.org/main/getting-started/installation.html)
- [PaddlePaddle 自定义算子（中文）](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html) —— 框架接入视角的对照阅读

## 8. 国产硬件

见 [npu-roadmap.md](npu-roadmap.md)（昇腾 Ascend C / Triton-Ascend / torch_npu、摩尔线程 MUSA、寒武纪 mlu-ops、海光 DCU、昆仑芯 XPU）。
