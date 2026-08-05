# 第一版实施决策

更新时间：2026-07-31

## 已确认

| 项目 | 选择 |
| --- | --- |
| 任务范围 | 单任务，`LIBERO-Spatial task 0` |
| 数据集 | `HuggingFaceVLA/libero` |
| 10 步教师 | 官方 LIBERO 预训练 checkpoint，具体仓库 ID 等待接口子决策 |
| 蒸馏范围 | 第一版只蒸馏动作专家 |
| 蒸馏路径 | 10 步教师到 5 步学生，再到 2 步学生 |
| ACT | `chunk_size=50`、`n_action_steps=10`、`n_obs_steps=1`、batch 16、训练 50k step |
| Diffusion Policy | horizon 64、`n_obs_steps=2`、`n_action_steps=10`、batch 8、训练 100k step |
| Diffusion Policy 推理消融 | 100、20、10、5 个去噪步 |
| 控制频率 | 20 Hz，每个控制周期 50 ms |
| chunk 重叠 | 10 个动作 |
| 夹爪处理 | 第 7 维采用迟滞切换 |
| RTC | 第一版启用 |
| RTC 首始参数 | overlap 10、guidance 10.0、指数衰减 |

配置的唯一来源是 `configs/libero_spatial_task0.toml`。实验脚本应读取该文件，避免多份参数逐渐分叉。

## 训练前硬门槛

1. 从 LIBERO suite 读取 task 0 的语言描述，再与 LeRobot 数据集的 `task_index` 和 episode 列表建立映射。
2. 检查教师 checkpoint 的输入接口。`lerobot/smolvla_libero` 配置记录 6 维状态和 3 路相机。`HuggingFaceVLA/smolvla_libero` 记录 8 维状态和 2 路相机，与新版数据集一致。需要在 4090 主机实际加载预处理器和一条样本，确认两个候选的加载与前向结果。
3. 使用同一条观测完成教师 10 步前向，检查输出形状为 `[1, 50, 7]`，数值有限，反归一化后落在 LIBERO 动作范围。
4. ACT 和 Diffusion Policy 各训练 100 step，随后完成一个仿真 episode。短流程只检查数据、模型、环境和评测链路能连通。
5. 根据数据和仿真各检查一次夹爪符号，再锁定迟滞的开合方向。阈值先用 `-0.5` 与 `0.5`，方向未确认时运行时必须拒绝启动。

## RTC 与手工重叠融合的边界

RTC 已经在 Flow Matching 采样过程中约束新 chunk 的重叠前缀。第一版保留一个动作队列，让控制线程和推理线程并行。SmolVLA 教师与蒸馏学生使用 RTC 做重叠连续性约束。ACT 和 Diffusion Policy 使用相同的队列与触发时机，并用普通重叠加权作为对照。

这样可以分别测量两类收益：

| 组件 | 主要影响指标 |
| --- | --- |
| 异步队列与提前触发 | 等待时间、实际控制频率、deadline miss |
| 普通重叠加权 | chunk 边界动作跳变、轨迹抖动 |
| RTC 采样引导 | Flow Matching 模型的边界连续性与任务成功率 |

## 仍由用户决定

正式长训练启动前，需要确定正式评测预算。当前开发评测固定为 1 个 seed，每个模型 20 个 episode。建议的正式配置是 3 个 seed，每个模型每个 seed 50 个 episode。这个选择会直接改变总仿真时长和实验置信度。

蒸馏损失权重当前暂设为 1:1。5 步学生短训后会报告两项损失的实际量级，再由用户决定是否调整。

教师仓库还需要一个子决策。当前建议 `HuggingFaceVLA/smolvla_libero`，因为输入契约与 `HuggingFaceVLA/libero` 完全一致。选择 `lerobot/smolvla_libero` 时需要额外确定 8 维状态到 6 维状态的变换以及第三路空相机来源。最终选择应依据 4090 上的真实前向测试。
