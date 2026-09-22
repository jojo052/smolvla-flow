# LIBERO 数据集、任务映射与评测协议

本项目把 LIBERO 的仿真评测任务和用于蒸馏的观测数据分开记录。二者共享任务语言和任务身份，但用途不同：仿真任务用于闭环成功率，观测数据用于教师监督下的动作专家蒸馏。

## 1. LIBERO 是什么

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 是面向终身机器人学习和操作策略迁移的仿真基准。任务由语言指令、场景、物体关系和目标条件共同定义，环境执行后通过任务成功状态判断回合是否完成。

本项目使用四个十任务套件：

| 套件 | 本项目中的含义 | 任务数 |
|---|---|---:|
| `libero_spatial` | 关注物体之间的空间关系和放置位置 | 10 |
| `libero_object` | 关注物体类别、物体选择和搬运 | 10 |
| `libero_goal` | 关注目标条件和目标状态 | 10 |
| `libero_10` | 多阶段、较长时程的操作任务 | 10 |

四套件共包含 40 个任务身份。每个任务由以下数据共同确定：

- 任务名称和自然语言指令
- BDDL 任务定义
- 初始状态文件
- 官方初始状态索引
- 任务最大控制步数
- 仿真环境成功判据

本项目没有把套件内的 `task_id` 当成数据集的 `task_index` 直接使用。运行入口根据任务名称和语言指令建立映射，再加载对应的 BDDL 与初始状态文件。

## 2. 本项目的正式仿真评测数据

正式评测固定使用每个任务官方初始状态索引 `0–9`：

```text
40 个任务 × 10 个初始状态 = 400 个回合 / 模型
```

评测协议如下：

| 项目 | 固定值 |
|---|---|
| 策略种子 | 123 |
| 环境种子 | 123 |
| 初始稳定阶段 | 10 个控制步，不计入动作预算 |
| Spatial 动作预算 | 220 tick |
| Object 动作预算 | 280 tick |
| Goal 动作预算 | 300 tick |
| LIBERO-10 动作预算 | 520 tick |
| 图像输入 | 两路 RGB，256×256 |
| 机器人状态 | 8 维 |
| 动作 | 7 维 |
| 动作块 | 50 个动作 |
| 主基准执行 | 同步、每执行 1 个动作重新预测 |
| RTC | 关闭 |
| 成功判据 | `info["is_success"]` |

每个模型均需完成 400 个回合。正常失败计入失败，不因失败重跑。环境初始化错误、CUDA 错误、文件缺失和非有限动作作为执行错误单独记录。

任务名称、语言指令、BDDL 哈希、初始状态文件哈希和状态哈希保存在：

[`artifacts/libero40_public_v1/manifest.json`](../artifacts/libero40_public_v1/manifest.json)

蒸馏学生的正式评测沿用同一任务和状态协议。T1、S1 的八环境交错执行只改变调度方式，成功率和每回合任务集合仍保持 40 任务、初始状态 `0–9`；其并行吞吐和服务延迟单独报告，不能与单环境吞吐直接比较。

## 3. 蒸馏使用的 `HuggingFaceVLA/libero` 数据

蒸馏阶段使用 `HuggingFaceVLA/libero` 的观测数据版本。当前记录如下：

```text
dataset: HuggingFaceVLA/libero
revision: 86958911c0f959db2bbbdb107eb3e17c5f9c798e
data manifest SHA-256: 9471cb4559782463642d593a6d82ec9a257ed3130e9f11dc40ec3966cf5bcf5e
```

当前实验记录声明：

- 40 个源任务指令与冻结的评测任务 manifest 一致。
- 源元数据声明约 1,693 个 episode 和 273,465 帧。
- 训练采样按任务均衡，避免 episode 数量较多的任务占据全部批次。
- episode 级边界用于训练和验证划分，避免同一 episode 的观测跨集合泄漏。
- 固定验证观测和固定噪声用于离线损失检查。
- S1 的 2→1 蒸馏使用冻结 T2 教师生成的动作目标。
- 专家动作不作为本轮 S1 的行为克隆损失；学生只接受蒸馏监督。

因此，蒸馏阶段的离线损失只能说明学生对教师目标的拟合程度。正式能力结论来自 LIBERO 仿真闭环回合。

## 4. 数据完整性和复现限制

历史源元数据中存在 episode 文件偏移与实际 Parquet footer 不一致的记录，包括文件 000 和文件 068。最终训练数据应以实际 Parquet footer、实际行索引和经过哈希确认的派生 manifest 为准，不能只依赖旧元数据中的偏移值。

归档时必须同时保存：

1. 数据 revision；
2. 数据 manifest 的 SHA-256；
3. 任务名称到数据任务的映射；
4. episode 划分和验证观测清单；
5. Parquet 文件 footer 审计结果；
6. 与训练入口对应的代码哈希。

如果公开归档不包含原始 episode 文件，报告仍保留 revision、manifest 哈希和下载说明。缺少实际训练 manifest 时，不把该归档称为完全可复现。

官方 `HuggingFaceVLA/smolvla_libero` checkpoint 的 revision、文件哈希、处理器哈希和输入输出契约记录在各运行 manifest 中。当前实验没有证明官方预训练数据与 LIBERO 蒸馏或评测数据完全无重叠，因此这一点作为研究限制保留。

## 5. 官方参考

- [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO 官方文档](https://lifelong-robot-learning.github.io/LIBERO/html/index.html)
- [HuggingFaceVLA/libero 数据集](https://huggingface.co/datasets/HuggingFaceVLA/libero)
- [LIBERO benchmark research page](https://libero-project.github.io/research)
