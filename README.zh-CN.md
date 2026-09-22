# SmolVLA Flow Matching：低步数蒸馏与实时动作块执行

本项目研究视觉语言动作策略在 LIBERO 闭环控制中的两个问题：

1. Flow Matching 的积分步数减少后，推理延迟和任务成功率如何变化？
2. 动作块异步执行是否比等待新动作块更适合实时控制，RTC 引导是否带来额外收益？

项目同时保留蒸馏训练、闭环回合、视频、权重与协议清单，结果表可以从归档的原始 JSON 重新生成。英文版见 [README.md](README.md)。

## 结论先行

- 在本项目固定的 40 任务、400 回合主基准上，官方 SmolVLA 两步推理 T2 达到 **74.50%**，是当前成功率最高的低步数 SmolVLA 配置。
- 五步蒸馏学生 S5 达到 **70.25%**，与官方权重直接五步推理 T5 相同。当前训练配方没有证明五步蒸馏能超过直接减步。
- 一步蒸馏学生 S1 达到 **71.25%**，比官方直接一步推理 T1 的 **62.50%** 高 8.75 个百分点，但仍比 T2 低 3.25 个百分点。这说明自蒸馏在一步边界有能力恢复作用，但不等于已经超过更高步数教师。
- 公共 ACT checkpoint 在同一 40 任务状态集合上为 **13.25%**，原生动作块吞吐为 **51.31 tick/s**。ACT 的动作块、语言条件和执行方式与 SmolVLA 不同，因此它是一个明确标注限制条件的外部基线，不应被解释成架构的普遍排名。
- S1 的 RTC 研究中，朴素异步执行比同步等待更可靠。RTC 能减少动作块边界的位移跳变，但本轮没有带来稳定的成功率提升。

## LIBERO 数据集与评测协议

### 四套件、40 个任务

主基准使用 LIBERO 的四个 10 任务套件：

| 套件 | 任务性质 | 任务数 |
| --- | --- | ---: |
| LIBERO-Spatial | 空间关系、位置和放置 | 10 |
| LIBERO-Object | 物体类别与物体操作 | 10 |
| LIBERO-Goal | 目标条件操作 | 10 |
| LIBERO-10 | 长程、多阶段操作 | 10 |

每个任务由任务名称、自然语言指令、BDDL 定义、官方初始状态文件、最大控制步数和 LIBERO 成功判据共同确定。冻结后的任务与初始状态哈希保存在 [40 任务 manifest](artifacts/libero40_public_v1/manifest.json) 中。

正式主基准显式使用每个任务官方初始状态索引 `0–9`：

```text
40 个任务 × 10 个初始状态 = 每个模型 400 个回合
```

固定协议如下：

- 策略随机种子和环境随机种子均为 `123`。
- 回合开始有 10 个稳定阶段控制步，不计入动作预算。
- Spatial、Object、Goal、LIBERO-10 的动作预算分别为 `220`、`280`、`300`、`520` 个控制步。
- 输入为两路 `256×256` RGB 图像和 8 维机器人状态。
- 输出动作是 7 维，动作块长度为 50。
- 主基准为同步闭环，关闭 RTC；每执行一个动作后重新预测。
- 成功由 LIBERO 环境返回的 `info["is_success"]` 判定。

完整的任务映射、初始状态、文件哈希和数据完整性说明见 [docs/libero_dataset.md](docs/libero_dataset.md)。官方参考包括 [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO)、[LIBERO documentation](https://lifelong-robot-learning.github.io/LIBERO/html/index.html) 和 [Hugging Face LIBERO dataset](https://huggingface.co/datasets/HuggingFaceVLA/libero)。

### 蒸馏训练数据与仿真评测数据的区别

蒸馏阶段使用 `HuggingFaceVLA/libero` 的观测、语言条件和 episode 边界，记录的 dataset revision 为：

```text
86958911c0f959db2bbbdb107eb3e17c5f9c798e
```

训练数据 manifest 的 SHA-256 为：

```text
9471cb4559782463642d593a6d82ec9a257ed3130e9f11dc40ec3966cf5bcf5e
```

蒸馏只使用冻结教师产生的动作目标。S1 的 2→1 训练没有加入专家动作行为克隆损失。训练数据用于学习教师动作，正式成功率来自 LIBERO MuJoCo 闭环回合，两者在报告中分开统计。部分历史元数据的 episode 偏移与 Parquet footer 存在不一致，因此归档时以实际 Parquet 行索引和 footer 审计结果为准。

## 40 任务主结果

每一行均为 400 个已完成回合，每个套件 100 个回合。T10、T5、S5、T2 使用历史单环境执行器；T1、S1 使用经过单独审计的八环境交错执行器，因此它们的并行吞吐与前四行的单环境吞吐分开解释。

| 模型 | 定义 | Spatial | Object | Goal | Long | 40 任务宏平均 | 预测 P50/P95 | rollout 吞吐 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T10 | 官方 checkpoint，10 步 | 72/100 | 90/100 | 79/100 | 34/100 | **68.75%** | 297.0 / 410.8 ms | 3.03 tick/s |
| T5 | 官方 checkpoint，直接 5 步 | 76/100 | 86/100 | 75/100 | 44/100 | **70.25%** | 162.7 / 222.1 ms | 5.33 tick/s |
| S5 | T10 → S5 蒸馏 | 76/100 | 93/100 | 75/100 | 37/100 | **70.25%** | 162.0 / 221.1 ms | 5.37 tick/s |
| T2 | 官方 checkpoint，直接 2 步 | 79/100 | 94/100 | 80/100 | 45/100 | **74.50%** | 92.5 / 123.8 ms | 8.55 tick/s |
| T1 | 官方 checkpoint，直接 1 步 | 73/100 | 65/100 | 72/100 | 40/100 | **62.50%** | 并行服务：69.4 / 91.6 ms | 并行：13.33 tick/s |
| S1 | T2 → S1 蒸馏 | 80/100 | 88/100 | 76/100 | 41/100 | **71.25%** | 并行服务：68.2 / 73.3 ms | 并行：13.86 tick/s |
| ACT | `jamongsteak/act_libero` 公共权重 | 25/100 | 11/100 | 3/100 | 14/100 | **13.25%** | 9.5 / 10.6 ms | 51.31 tick/s |

### 蒸馏结果怎么解释

- **S5 对 T5**：宏平均差异为 0 个百分点。S5 在 Object 高 7 个百分点，在 Long 低 7 个百分点，Spatial 和 Goal 相同。它证明了当前训练能够在五步预算下学习教师行为，但没有证明蒸馏优于直接五步积分。
- **S1 对 T1**：宏平均提高 8.75 个百分点，Spatial、Object、Goal、Long 分别提高 7、23、4、1 个百分点。一步蒸馏恢复了直接减步损失的一部分。
- **S1 对 T2**：宏平均低 3.25 个百分点。S1 是一步学生，T2 是两步官方权重，二者回答的是不同的能力与计算预算问题。
- 当前主基准下，T2 是低步数 SmolVLA 中成功率最高的配置；蒸馏的主要价值是让更低积分预算的学生恢复教师能力，而不是保证超过直接减步。

T1 与 S1 的服务延迟和吞吐来自并行执行器，不能直接填入 T10、T5、S5、T2 的单环境延迟列。T1 的并行吞吐为 13.33 tick/s，S1 为 13.86 tick/s；这两个数字是批量调度后的端到端合计吞吐，不是单请求延迟除以 batch。

## ACT 外部基线

ACT 使用公开的 `jamongsteak/act_libero` checkpoint，revision 为 `8744e679f4038386070ff70723edab9bb6228bd3`，模型文件 SHA-256 为：

```text
ae1d034c4d30879773a1f67590194a7ce5246c8466eb46db299696bfab2bb247
```

ACT 在同一 40 个任务和同一组 `0–9` 初始状态上完成 400 个回合，得到 53/400，宏平均为 13.25%。在其原生配置下，ACT 预测并执行 100 个动作组成的 chunk，不使用 temporal ensemble，也没有语言条件。SmolVLA 主基准则每执行一个动作重新预测，动作块长度虽同为 50，但执行语义不同。因此 ACT 的 9.5 ms 预测和 51.31 tick/s 只能作为原生部署速度参考，不能和 SmolVLA 的单动作重预测延迟做严格同口径排名。

T10 与 ACT 在相同任务和初始状态上的配对结果为：T10 单独成功 232 回合，ACT 单独成功 10 回合，双方都成功 43 回合，双方都失败 115 回合。配对成功率差为 T10−ACT = 55.5 个百分点，按任务 bootstrap 的 95% 区间为 [44.5, 65.5] 个百分点。完整的 40 行逐任务表在 [artifacts/libero40_comparison_v1/results.md](artifacts/libero40_comparison_v1/results.md)。历史 task34 的 ACT 30-seed 结果不并入这张主表，见 [docs/legacy_task34.md](docs/legacy_task34.md)。

这个比较可以回答“当前选定公开 checkpoint 在同一 LIBERO 状态集合上的结果如何”，不能单独回答“ACT 架构是否普遍优于 SmolVLA”。两者训练数据、语言条件、动作块执行语义和公开权重来源并不完全相同。

## RTC 与异步动作块结果

RTC 研究单独使用 S1，在 8 个任务、初始状态 `20–24` 上进行，每个条件 40 个回合。它不是主基准新增的 400 回合。RTC 固定引导强度为 1.25；同步和朴素异步结果作为同一研究中的对照。

### 自然推理延迟

| 执行方式 | 成功/40 | 成功率 | 平均回合耗时 | 成功回合/小时 | 队列耗尽率 | 预测 P50/P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 同步动作块 | 13/40 | 32.50% | 18.75 s | 62.40 | 22.69% | 102.42 / 126.17 ms |
| 朴素异步 | **33/40** | **82.50%** | 14.17 s | **209.60** | 0% | 101.79 / 118.33 ms |
| 异步 + RTC 1.25 | 24/40 | 60.00% | 15.20 s | 142.08 | 0% | 108.40 / 120.58 ms |

### 额外注入 100 ms 延迟

| 执行方式 | 成功/40 | 成功率 | 平均回合耗时 | 成功回合/小时 | 队列耗尽率 | 预测 P50/P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 同步动作块 | 3/40 | 7.50% | 20.65 s | 13.08 | 32.69% | 109.68 / 130.01 ms |
| 朴素异步 | **27/40** | **67.50%** | 15.42 s | 157.57 | 0% | 114.43 / 132.34 ms |
| 异步 + RTC 1.25 | **27/40** | **67.50%** | 14.45 s | 168.11 | 0% | 117.99 / 130.94 ms |

RTC 将自然延迟下的平均平移动作块边界跳变从 `0.26093` 降到 `0.10666`，在额外 100 ms 延迟下从 `0.28742` 降到 `0.11170`。因此本轮可以支持“RTC 改善动作衔接平滑性”的结论，但不能支持“RTC 稳定提高成功率”。完整分析见 [docs/final_results.md](docs/final_results.md) 和 [docs/rtc_weight125_results.md](docs/rtc_weight125_results.md)。

## 复现、归档与限制

主要复现入口和报告：

- 主协议：[configs/libero40_public_v1.json](configs/libero40_public_v1.json)
- LIBERO 数据与完整性说明：[docs/libero_dataset.md](docs/libero_dataset.md)
- S1 蒸馏与选模记录：[docs/s1_execution.md](docs/s1_execution.md)
- S1 checkpoint：[artifacts/distill40_s1_formal_v1/training/010000.pt](artifacts/distill40_s1_formal_v1/training/010000.pt)
- 40 任务主结果：[docs/final_results.md](docs/final_results.md)
- RTC 1.25 结果：[docs/rtc_weight125_results.md](docs/rtc_weight125_results.md)
- ACT 审计与逐任务对照：[artifacts/libero40_act_v1/](artifacts/libero40_act_v1/)、[artifacts/libero40_comparison_v1/](artifacts/libero40_comparison_v1/)
- 旧 task34 实验附录：[docs/legacy_task34.md](docs/legacy_task34.md)

外置归档目录为：

```text
/Volumes/DataLabWork/projects/smolVLA-flow/project-final-2026-09-20/
```

归档包含 README、协议、原始回合 JSON、视频、manifest、选定 S1 checkpoint 和 `archive_manifest.sha256`。没有复制凭据、Python 环境、模型缓存和数据缓存。官方 SmolVLA 预训练数据与 LIBERO 的潜在重叠信息不完整，因此不宣称二者完全无数据重叠。正式状态 `0–9` 是固定对照集合，不称为未接触测试条件上的最终泛化证明。

在本地安装开发依赖后可运行：

```bash
python -m pip install -e ".[dev,plot]"
python -m pytest -q
```
