# 国产后端路线图（NPU Roadmap）

> 策略：**先以 CUDA 为主打透（001-093），题面保持硬件无关**，昇腾等国产后端从第11章起逐步接入。
> 原则：同一道题的多种后端放在 `solutions/<backend>/` 下并列展示，测试自动发现可用后端、缺硬件则跳过——贡献者不需要拥有所有硬件也能出题/做题。

## 后端目录约定

```
solutions/
├── cuda/        # NVIDIA CUDA C++
├── triton/      # OpenAI Triton（NVIDIA，经 triton-ascend 亦可跑昇腾）
└── npu_ascendc/ # 昇腾 Ascend C（预留，第11章起）
```

每个后端目录统一暴露 `solution.py`，导出与 `reference.py` 同名的函数；`common/ops100/testing.load_problem` 会自动扫描。

## 阶段规划

### P0（已完成）：CUDA + Triton 打基础
001-008 已双后端实现。这一步建立「题面-参考实现-多后端-自动评测」的完整闭环。

### P1（第11章 093）：HIP/ROCm 移植
- CUDA 与 HIP 语法高度同源（`cudaMalloc` → `hipMalloc`），`hipify-perl` 可自动转换大部分 kernel；
- 海光 DCU 基于 ROCm/DTK 生态，是 CUDA 学习者最低成本的国产化路径；
- 出口：093 题给出 CUDA→HIP 的移植对照表与脚本。

### P2（第11章 094-097）：昇腾优先接入
- **Ascend C**（华为官方算子开发语言，CANN 生态）：
  - 官方指南：《[Ascend C 算子开发指南](https://www.hiascend.com/document/detail/zh/CANNCommercial/810/opdevg/ascendcopdevg/atlas_ascendc_10_0001.html)》
  - 官方样例：CANN 生态仓库 [gitcode.com/cann](https://gitcode.com/cann)（原 gitee.com/ascend/samples 已停止维护，样例迁至此）
  - 算子共建库：[gitee.com/ascend/cann-ops](https://gitee.com/ascend/cann-ops)（注意其许可仅限昇腾硬件使用）
  - 规划题：094 向量加法、095 RMSNorm、096 FlashAttention 精简版（Cube/Vector 分工是其核心学习点）
- **Triton-Ascend**（同一份 Triton 代码跑昇腾，学习成本最低的 NPU 路径）：
  - 仓库：[gitcode.com/Ascend/triton-ascend](https://gitcode.com/Ascend/triton-ascend)（原 gitee 镜像，MIT）
  - 官方已验证 VectorAdd/Softmax/LayerNorm/FlashAttention/Matmul 示例——与本题库 001/006/008/043 一一对应
  - 规划题：097 用 Triton-Ascend 跑同一份 softmax
- **torch_npu**（PyTorch 昇腾适配，BSD）：
  - 仓库：[gitee.com/ascend/pytorch](https://gitee.com/ascend/pytorch)
  - 自定义算子接入文档：[华为官方·PyTorch 算子开发](https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/Frameworkfeatures/featuresguide_00021.html)

### P3：摩尔线程 MUSA
- 编程指南：[docs.mthreads.com MUSA SDK Programming Guide](https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/)（SIMT 模型与 CUDA 高度相似，含 TME、GEMM/FA 优化章节）
- 官方教程仓库：[MooreThreads/tutorial_on_musa](https://github.com/MooreThreads/tutorial_on_musa)；PyTorch 适配 [torch_musa](https://github.com/MooreThreads/torch_musa)

### P4：其他国产平台
- **寒武纪**：BANG C 文档见[寒武纪开发者社区](https://developer.cambricon.com/index/document/index/classid/3.html)（⚠️ 部分资源对企业开放）；开源算子参考 [Cambricon/mlu-ops](https://github.com/Cambricon/mlu-ops)
- **海光 DCU**：DTK/ROCm 生态，见[光合开发者社区](https://www.sourcefind.cn/)（⚠️ 文档站为动态加载）
- **昆仑芯 XPU**：公开文档较少；算子实现可读 Paddle 仓 `paddle/phi/kernels/xpu/`（[示例](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/kernels/xpu)）

## 接入的技术约定（写给我们自己）

1. **测试发现**：`load_problem` 扫描 `solutions/*/solution.py`，新后端零侵入接入；
2. **环境探测**：昇腾用 `torch.npu.is_available()`（torch_npu）、MUSA 用 `torch_musa`、HIP 用 `torch.version.hip`，测试按平台 skip；
3. **正确性基准不变**：所有后端与 PyTorch 参考实现对齐（跑在各自设备上的 reference）；
4. **性能口径不变**：bytes/flops 公式硬件无关，bandwidth 用各硬件标称带宽归一化报告「带宽利用率 %」；
5. **文档双语**：昇腾题目 README 用中文为主（目标读者明确）。
