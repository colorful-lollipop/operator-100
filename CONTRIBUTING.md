# 贡献指南（CONTRIBUTING）

感谢参与！本项目的一切建设都围绕一个目标：**让每个题目目录都是一个自洽的、可运行的、有出处的教学单元**。

## 如何认领一道题

1. 在 [CURRICULUM.md](CURRICULUM.md) 中找到 🔲 待实现题目，开 issue 或直接在 PR 中声明认领；
2. 按 [docs/problem-template.md](docs/problem-template.md) 出题；
3. 完成后把 CURRICULUM.md 中状态改为 ✅，并在 PR 里贴上 `test.py` 与 `bench.py` 的输出（注明 GPU 型号与环境，见 [docs/benchmark.md](docs/benchmark.md) §5）。

## 如何为已有题目补充后端解法

- 新增 `problems/NNN-*/solutions/<backend>/solution.py`（现有后端：`cuda`、`triton`）；
- 函数名/签名必须与 `reference.py` 一致；
- 无需改 `test.py`——`common.ops100.testing.load_problem` 会自动发现新后端；如后端需要额外环境判断，在 `solution.py` 顶部抛出带清晰信息的 ImportError 即可（会被捕获并跳过）。

## 代码风格

- Python：ruff（`ruff check .`），行内中文注释为主，命名 snake_case；
- CUDA/C++：kernel 名 snake_case，host 函数与 kernel 用 `__global__`/`__device__` 明确区分；每个 kernel 文件头部有中文总注释（做什么 + 关键设计）；
- 提交信息：`[001] feat: grid-stride 版本` / `[043] fix: fp16 精度容差` / `docs: 更新评测规范`。

## 质量底线（PR 必查）

1. 正确性优先于性能：不确定的优化宁可注释掉，也不能让 test 变红；
2. 所有 kernel 有边界保护、binding 有 TORCH_CHECK；
3. 不引入重量级依赖；示例代码可运行、可复现；
4. 尊重引用来源：题目出处必须链接到原论文/原仓库，杜绝整段搬运未授权内容——**我们的题面必须是原创叙述，参考实现必须是自己写的**。

## 环境

- 本地开发：见 README「快速开始」；Windows 下 Triton 可用 `triton-windows` 社区轮子；
- CI：CPU 环境跑 lint + 参考实现测试；GPU 测试在本地完成后贴结果。
