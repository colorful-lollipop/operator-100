# 出题模板（Problem Template）

> 复制本目录结构出题。命名：`problems/<三位题号>-<短横线小写名>/`，如 `problems/009-embedding-lookup/`。
> 题面质量标准：**一个聪明的本科生读完 README 不需要再问"这题让干嘛"**。

## 目录结构

```
problems/NNN-name/
├── README.md               # 题面（按下述结构写）
├── reference.py            # PyTorch 参考实现（正确性基准；导出与题名一致的函数）
├── test.py                 # 正确性测试（自动发现可用后端，缺依赖跳过）
├── bench.py                # 性能基准（必须给出 bytes/flops 公式）
└── solutions/
    ├── cuda/
    │   ├── solution.cu     # kernel + PYBIND11_MODULE
    │   └── solution.py     # torch.utils.cpp_extension.load 入口，导出同名函数
    ├── triton/
    │   └── solution.py     # 导出同名函数
    └── npu_ascendc/        # （可选）昇腾后端，见 docs/npu-roadmap.md
        └── solution.py
```

## README.md 结构

```markdown
# NNN · 题目名（中文名）

难度：★~★★★★★ | 章节：第X章 | 标签：tag1、tag2

## 题目
一段话讲清楚：输入是什么、输出是什么、语义是什么。
公式用 LaTeX 行内式，如 $y = x \cdot w$。

## 函数签名
def op(x: torch.Tensor, ...) -> torch.Tensor

## 输入约束
- dtype / shape / 连续性 / 取值范围，逐条列出

## 要求
必做：
1. （正确性硬性要求）
2. （接口/实现方式要求）

选做（进阶）：
1. （性能目标 / 变体 / 分析题）

## 提示
<details><summary>提示 1：...</summary>...</details>   # 递进式 2-4 条

## 常见陷阱
- 列出 2-4 个真实会踩的坑（越界、精度、同步……）

## 评分标准
- 正确性：python test.py 全绿（给出容差）
- 性能：给出达标线，如「≥ torch eager 的 X%」或「≥ Y GB/s」

## 参考资料
- 论文 / 仓库 / 博客链接（与 CURRICULUM.md 出处一致）
```

## 代码规范

1. **接口一致**：所有后端导出与 `reference.py` 同名同参的函数；
2. **入口统一**：CUDA 方案用 `torch.utils.cpp_extension.load`，JIT 模块名必须唯一（`ops100_<题号>_<算子名>`）；
3. **健壮性**：binding 里 `TORCH_CHECK` 校验设备/dtype/形状；kernel 必须有边界保护；大索引用 `long`；
4. **Windows/中文系统兼容（重要）**：`.cu` 文件里**字符串字面量一律用英文**（TORCH_CHECK 报错信息、pybind docstring）——中文 Windows（GBK 代码页）下 nvcc 无法解析 UTF-8 字符串字面量，会报 "missing closing quote"；**中文注释是安全的**（多字节序列不含换行），可保留；
5. **注释**：kernel 文件头部用中文注释讲清「这个 kernel 在做什么、关键设计为什么这样」，注释密度向教学倾斜；
6. **测试**：至少 3 组形状（含非整除边界）+ fp32 必测 + 有 fp16 路径的加一组 fp16；容差参考 docs/benchmark.md；
7. **基准**：`bench.py` 对比 torch eager，并打印 GB/s 或 TFLOPS（公式注释写明）；
8. **同一性**：每个 `solution.py` 第一行注释说明「本文件是题 NNN 的 XX 后端实现」。

## 自查清单（PR 前逐项打勾）

- [ ] 目录/文件命名符合规范，题号未与现有题目冲突
- [ ] `python test.py` 在有 GPU 机器全绿；CPU 机器上自动跳过后端不报错
- [ ] `python bench.py` 输出含 bytes/flops 口径的带宽/算力
- [ ] README 齐全且链接可用；CURRICULUM.md 中该题状态已更新为 ✅
- [ ] CUDA kernel 有边界保护；binding 有 TORCH_CHECK；无 int 溢出隐患
- [ ] 新增依赖（如有）在 README 环境表和 pyproject 中注明
