# `current_seed123` 旧版评测产物

> 状态：pre-fix，deprecated。该目录只保留用于审计早期实验和定位评测口径变化，请勿将其验收结果用于项目结论。

这批产物使用 LIBERO Spatial task 0，对应数据集 task index 34。环境 seed 为 0 到 4，`torch_seed=123`。JSON 数值保持不变，机器绝对路径已替换为占位符；日志只在本地保留，不进入 Git。该目录用于与修正后的 `current_seed123_v2` 逐项对照。

## 已知评测问题

1. `effective_control_hz` 的计时包含 episode 结束后的推理线程收尾时间（finalization），无法准确表示控制循环频率。
2. `chunk_boundary_jump` 每固定 10 个动作取一个边界，没有根据动作队列实际的 chunk 切换位置计算。
3. 同步与异步 rollout 的夹爪动作后处理不一致，成功率和平滑度缺少可比性。
4. `task_isolation` 只读取 rollout 中记录的 task index，缺少蒸馏数据过滤证据时仍会误报 `pass`。

## 文件用途

- `acceptance_metrics.json`：旧评测器产生的汇总，只用于对照。
- `native_teacher_sync.json`：原生 10 步 SmolVLA 同步 rollout。
- `undistilled_two_step_sync.json`：未蒸馏 2 步 SmolVLA 同步 rollout。
- `distilled_two_step_sync.json`：蒸馏 2 步 SmolVLA 同步 rollout。
- `distilled_two_step_async_rtc.json`：蒸馏 2 步 SmolVLA 异步 RTC rollout。
- `distilled_two_step_async_unfused.json`：蒸馏 2 步 SmolVLA 异步无融合对照。
- `run.log`：当次运行日志，只在执行机器和本地工作区保留。

修正后可用的评测产物位于 `../current_seed123_v2/`。
