# operator-100 · 深度学习算子 100 题

> 以大模型算子为主线，从「向量加法」一路写到「Flash Attention」的动手题库。
> 每题多种解法：**PyTorch 参考 / CUDA / Triton**，后续接入**昇腾（Ascend C / Triton-Ascend）**等国产后端。

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)

---

## 这是什么

会调 PyTorch 不等于会写高性能算子。大模型时代，**算子开发是推理/训练性能优化的核心技能**——FlashAttention、PagedAttention、Fused MoE 这些工作全部始于手写 kernel。本项目用 LeetCode 的形式把它拆成 100 道题：

- **题目分层**：11 章 × 100 题，从 ★ 入门（向量加法）到 ★★★★★ 专家（FlashAttention-3 风格流水线），每题 30 分钟 ~ 数天不等；
- **多解法对照**：每题提供 PyTorch 参考实现（正确性基准），CUDA 与 Triton 两套手写方案，站在同一测试与基准口径下对比；
- **自动评测**：每题自带 `test.py`（正确性，自动跳过缺硬件的后端）与 `bench.py`（带宽 / TFLOPS / 对比 PyTorch 的加速比）；
- **有出处的题库**：每题标注参考论文 / 开源实现 / 经典博客（KernelBench、Triton 官方教程、vLLM、Liger-Kernel、FlashAttention、llm.c……），见 [docs/references.md](docs/references.md)；
- **预留国产后端**：题面与硬件无关，解法目录按后端约定扩展，昇腾路线图见 [docs/npu-roadmap.md](docs/npu-roadmap.md)。

完整题目清单与知识点映射：**[CURRICULUM.md](CURRICULUM.md)**

## 快速开始

### 环境

| 组件 | 要求 | 说明 |
|---|---|---|
| GPU | NVIDIA，架构 ≥ Maxwell | 评测在 RTX 4070 Ti SUPER 上校准 |
| CUDA Toolkit | ≥ 12.x | `nvcc --version` 可用；与 PyTorch 的 CUDA 大版本一致 |
| PyTorch | ≥ 2.1（CUDA 版） | `torch.cuda.is_available()` 为 True |
| 编译器 | Windows 需 MSVC（Visual Studio）；Linux 需 gcc | torch cpp_extension 编译 CUDA 算子用 |
| Triton | ≥ 3.0 | 可选，Triton 方案用；Windows 可用社区轮子 `triton-windows` 或 WSL2 |
| Python | ≥ 3.10 | 推荐 conda 环境 |

```bash
pip install "torch>=2.1" pytest ninja      # Linux/WSL: 加 triton
```

### 跑通第一题

```bash
cd problems/001-vector-add
python test.py        # 正确性：torch / cuda / triton 三种实现 vs 参考答案
python bench.py       # 性能：GB/s 对比（memory-bound 算子看带宽）
```

首次运行 CUDA 方案会自动 JIT 编译（1~2 分钟），之后走缓存。

### Windows 用户注意

- 找不到 `cl.exe` / `ninja`：用仓库自带脚本 `tools\run_with_msvc.bat <命令>`，它会自动加载 MSVC 环境并把 Python 环境目录加入 PATH；
- VS 版本过新（如 VS 2026）：当前 nvcc 可能报 `unsupported Microsoft Visual Studio version` 或 `cudafe++ died`，要么升级 CUDA Toolkit 到与之匹配的版本，要么在 `solution.py` 的 `extra_cuda_cflags` 保留 `-allow-unsupported-compiler`（本项目已内置）；
- **中文 Windows 的经典坑**：`.cu` 文件的字符串字面量（如 TORCH_CHECK 报错信息）必须用英文——GBK 代码页下 nvcc 解析 UTF-8 字符串会报 "missing closing quote"；中文注释则安全无碍；
- Triton：直接 `pip install triton-windows`（社区轮子）即可在 Windows 原生运行，无需 WSL2。

### 题目目录结构（每题自包含）

```
problems/001-vector-add/
├── README.md               # 题面：描述 / 签名 / 要求 / 提示 / 评分标准 / 参考
├── reference.py            # PyTorch 参考实现（正确性基准）
├── test.py                 # 正确性测试：python test.py 或 pytest 统一收集
├── bench.py                # 性能基准：耗时 / GB/s / TFLOPS
└── solutions/
    ├── cuda/
    │   ├── solution.cu     # CUDA kernel + pybind 绑定
    │   └── solution.py     # torch.utils.cpp_extension.load 加载入口
    └── triton/
        └── solution.py     # Triton kernel
```

约定：

1. **接口一致**：各后端 `solution.py` 必须导出与 `reference.py` 同名的函数；
2. **测试自动适配**：`test.py` 自动发现可用后端，缺硬件/缺依赖时跳过（CI 的 CPU 环境只跑参考实现）；
3. **性能即考点**：`bench.py` 必须给出 bytes / flops 计算公式（roofline 口径见 [docs/benchmark.md](docs/benchmark.md)）。

## 100 题路线图

| 章节 | 题号 | 主题 | 难度 |
|---|---|---|---|
| 第1章 入门 | 001-008 | 元素级算子、归约、转置、softmax、RMSNorm/LayerNorm | ★ |
| 第2章 内存与索引 | 009-016 | gather/scatter、直方图、扫描、Top-K、变长打包 | ★★ |
| 第3章 GEMM | 017-026 | 朴素→分块→寄存器→双缓冲→Tensor Core→epilogue 融合 | ★★★ |
| 第4章 LLM 融合算子 | 027-036 | 残差融合、SwiGLU、online softmax、交叉熵、dropout | ★★~★★★★ |
| 第5章 RoPE | 037-040 | 两种 RoPE 布局、QKV 融合、ALiBi | ★★~★★★ |
| 第6章 Attention | 041-052 | 朴素→FlashAttention v1/v2/GQA/bwd→PagedAttention→投机验证 | ★★★~★★★★★ |
| 第7章 量化 | 053-060 | INT8/INT4/FP8、GPTQ/AWQ/SmoothQuant、KV Cache 量化 | ★★★~★★★★ |
| 第8章 MoE | 061-068 | 路由、permute/unpermute、grouped GEMM、shared expert | ★★★~★★★★ |
| 第9章 解码与采样 | 069-078 | argmax/top-k/top-p、KV 分页管理、flash-decoding | ★★~★★★★ |
| 第10章 训练侧 | 079-086 | 各算子 backward、融合 AdamW、梯度裁剪 | ★★★~★★★★ |
| 第11章 系统与毕业设计 | 087-100 | CUDA Graph、持久化 kernel、昇腾移植、迷你推理引擎 | ★★★~★★★★★ |

当前进度：**001-008（第1章全部）已实现**，可作全套模板参考。剩余题目按 [CURRICULUM.md](CURRICULUM.md) 持续建设，欢迎按 [CONTRIBUTING.md](CONTRIBUTING.md) 认领。

## 学习路径建议

- **零基础**：第1章 → 第2章，配合 PMPP 教材前 10 章与 [gpu-mode/lectures](https://github.com/gpu-mode/lectures) L1-L9；
- **有 CUDA 基础**：直接第3章 GEMM（对照 [siboehm 的 matmul 优化博客](https://siboehm.com/articles/22/CUDA-MMM)），然后第4、6章；
- **冲大模型推理**：第6章 Attention + 第7章量化 + 第9章解码，对照阅读 vLLM 与 FlashAttention 源码；
- **训练框架方向**：第10章 backward 系列 + Liger-Kernel 源码；
- **国产硬件**：第11章 094-097（昇腾 Ascend C / Triton-Ascend），见 [docs/npu-roadmap.md](docs/npu-roadmap.md)。

## 文档

- [CURRICULUM.md](CURRICULUM.md) —— 100 题总览（题面一句话 + 知识点 + 出处）
- [docs/problem-template.md](docs/problem-template.md) —— 出题模板
- [docs/benchmark.md](docs/benchmark.md) —— 评测规范（计时 / 容差 / roofline）
- [docs/references.md](docs/references.md) —— 参考资源地图（论文 / 仓库 / 博客 / 课程）
- [docs/npu-roadmap.md](docs/npu-roadmap.md) —— 昇腾等国产后端接入路线
- [CONTRIBUTING.md](CONTRIBUTING.md) —— 贡献指南

## 致谢

题库设计与资料大量参考了这些优秀的开源项目与文章：KernelBench（Stanford）、OpenAI Triton Tutorials、GPU MODE（原 CUDA-MODE）、FlashAttention、vLLM、Liger-Kernel、FlagGems、llm.c、《PMPP》、NVIDIA Developer Blog、OneFlow 技术博客、陈子旸《AI系统》等，详见 [docs/references.md](docs/references.md)。

## License

[MIT](LICENSE)
