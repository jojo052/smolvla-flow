# `current_seed123_v2` 验收评测产物

本目录对应 commit `8e765b3`。实验包含 5 个模型或控制配置，每组运行 5 个 episode。环境 seed 统一为 0 到 4，`torch_seed=123`。LIBERO Spatial task 0 对应数据集 task index 34。

## 五组对照

| 配置 | 采样步数 | 控制方式 | 成功数 | 成功率 | 平均有效控制频率 |
| --- | ---: | --- | ---: | ---: | ---: |
| 原生 SmolVLA teacher | 10 | 同步 | 2/5 | 40% | 2.275 Hz |
| 未蒸馏 SmolVLA | 2 | 同步 | 2/5 | 40% | 4.501 Hz |
| 蒸馏 SmolVLA | 2 | 同步 | 2/5 | 40% | 4.491 Hz |
| 蒸馏 SmolVLA + 异步 RTC | 2 | 异步 | 2/5 | 40% | 10.326 Hz |
| 蒸馏 SmolVLA + 异步无融合 | 2 | 异步 | 3/5 | 60% | 10.093 Hz |

`effective_control_hz` 使用控制循环本身的时间计算，不包含 episode 结束后的 finalization。同步和异步配置统一使用夹爪迟滞切换。异步 chunk 边界按动作队列的 sequence transition 识别。

## 11 项验收结果

| 指标 | 状态 | 核心数值或说明 |
| --- | --- | --- |
| finite output | pass | rollout 有限值比例最低为 100%，benchmark 输出均为 finite |
| task isolation | unknown | rollout 中观测到 task index 34，现有蒸馏产物缺少 task filter 证据 |
| waiting ticks | pass | 2159 个控制步中 waiting tick 为 0 |
| deadline miss ratio | pass | 25 次异步推理事件中 0 次 deadline miss，比例 0% |
| effective control frequency | fail | 最低 8.525 Hz，平均 10.326 Hz，目标为至少 15 Hz |
| two-step speedup | pass | 2 步学生相对 10 步 teacher 加速 3.102 倍，目标为至少 2.5 倍 |
| async success drop | pass | 蒸馏模型同步和异步 RTC 成功率均为 40%，下降 0 个百分点 |
| validation action error | unknown | 缺少蒸馏与未蒸馏 2 步模型的同一 held-out 验证集动作误差 |
| chunk boundary jump | pass | RTC 融合为 0.2851，无融合为 0.6353 |
| action smoothness | fail | RTC 融合为 0.04332，无融合为 0.04025，目标为融合值更低 |
| same episodes and seeds | pass | 两组用于对照的 rollout 均使用 `(seed, torch_seed) = (0..4, 123)` |

## 15 Hz 失败原因

异步 RTC 的单步 LIBERO `env.step()` 平均耗时在 86.36 ms 到 115.27 ms 之间。这部分耗时已经高于 15 Hz 对应的 66.67 ms 周期，控制频率因此落在 8.52 Hz 到 11.33 Hz。异步队列没有等待 tick，推理事件也没有 deadline miss。当前瓶颈记录在仿真环境步进端。

## 结果边界

`task_isolation` 保持 `unknown`，直到蒸馏数据产物记录 `task_index_filter`、过滤前后样本数，并能证明混入其他 task 的样本数为 0。

`validation_action_error` 保持 `unknown`，直到在同一 held-out 验证集、同一噪声和同一条件下，完成蒸馏 2 步模型与未蒸馏 2 步模型的动作误差对照。

`chunk_boundary_jump` 已达标，RTC 重叠融合对真实 chunk 切换点的跳变有明显降低。`action_smoothness` 使用整条动作轨迹的相邻差分，本轮 RTC 值高于无融合对照，所以保持 `fail`。

## 文件用途

- `acceptance_metrics.json`：11 项验收指标汇总。
- `native_teacher_sync.json`：原生 10 步 teacher 同步 rollout。
- `undistilled_two_step_sync.json`：未蒸馏 2 步模型同步 rollout。
- `distilled_two_step_sync.json`：蒸馏 2 步模型同步 rollout。
- `distilled_two_step_async_rtc.json`：蒸馏 2 步模型异步 RTC rollout。
- `distilled_two_step_async_unfused.json`：蒸馏 2 步模型异步无融合对照。
