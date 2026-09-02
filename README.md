# SmolVLA Flow-Matching 动作专家蒸馏与异步闭环优化

本项目在统一的 LIBERO task34 协议下，对比 ACT、Diffusion Policy、原生 SmolVLA、未蒸馏 2-step SmolVLA 和开发版蒸馏 2-step SmolVLA。

五种策略的 30-seed 正式同步 rollout 已完成。ACT 在当前任务上取得最高成功率和最高完整 rollout 吞吐。SmolVLA 从 10 个 Flow steps 降到 2 个后获得明显加速，当前开发版蒸馏适配器没有恢复成功率。

## 正式实验

| 项目 | 设置 |
| --- | --- |
| 任务 | LIBERO-Spatial task 0，对应数据集 task_index 34 |
| 环境 seed | 0 到 29，每种策略 30 个 episode |
| 策略采样 seed | 123 |
| 观测 | 两路 256 × 256 图像和 8D state |
| 动作 | 7D action，每个 chunk 执行 10 步 |
| episode 上限 | 280 个仿真步 |
| 夹爪方向 | positive_open |
| 运行模式 | 同步，RTC 关闭 |
| 成功判据 | LIBERO info 中的 is_success |

完整协议见 [configs/evaluation_protocol.toml](configs/evaluation_protocol.toml)。

## 正式结果

| 策略 | 成功率 | Wilson 95% 区间 | 完整 rollout 吞吐 | 混合 select_action 平均耗时 |
| --- | ---: | ---: | ---: | ---: |
| ACT 050000 | **29/30，96.67%** | 83.33% 到 99.41% | **19.99 tick/s** | **1.76 ms** |
| Diffusion 080000 | 25/30，83.33% | 66.44% 到 92.66% | 12.26 tick/s | 40.93 ms |
| 原生 SmolVLA，10 steps | 22/30，73.33% | 55.55% 到 85.82% | 13.32 tick/s | 36.26 ms |
| 未蒸馏 SmolVLA，2 steps | 21/30，70.00% | 52.12% 到 83.34% | 19.31 tick/s | 15.04 ms |
| 开发版蒸馏 SmolVLA，2 steps | 20/30，66.67% | 48.78% 到 80.77% | 19.27 tick/s | 14.97 ms |

所有结果精确使用相同的 30 个环境 seed 和策略 seed 123。全部动作均为有限值，所有失败 episode 均运行到 280 步上限。

## 结论与边界

- ACT 是当前 task34 的首选策略。ACT 对原生、未蒸馏 2-step 和开发版蒸馏 2-step SmolVLA 的 Holm 校正 p 分别为 0.03125、0.02344 和 0.01563。
- ACT 对 Diffusion 的观察成功率高 13.33 个百分点，双侧 exact McNemar p 为 0.125。30 个 seed 还不足以确认显著差异。
- 未蒸馏 2-step SmolVLA 相比原生 10-step，混合调用耗时降低约 58.5%，完整 rollout 吞吐提升约 45%，成功数减少 1/30。
- 开发版蒸馏 2-step 为 20/30，未蒸馏 2-step 为 21/30。当前蒸馏权重没有表现出质量恢复，完整 rollout 吞吐也未提升。
- 开发版适配器只使用 8 个样本和 10 次优化更新，不能支持正式 task34 蒸馏完成的结论。
- 混合 select_action 耗时包含动作 chunk 生成和动作队列弹出，不等同于单次完整模型 forward latency。

ACT 与 Diffusion 使用 task34-only 数据训练。SmolVLA 使用官方预训练 checkpoint，没有接受对等的 task34 训练预算。当前结果适合用于 task34 的 checkpoint 级部署选择。正式异步多 seed 矩阵仍未完成。

## Checkpoint

- ACT 使用 task34 训练的 050000 checkpoint。
- Diffusion 完成 100000 步训练。080000 在 held-out action validation 中取得最低 action MSE，因此用于正式 rollout。
- SmolVLA 使用 HuggingFaceVLA/smolvla_libero，snapshot 为 6721902bc4d61e50a3bfdb11dfb4cb626f05d102。

## 结果与复核

- [正式结果说明](artifacts/rollout/formal_seed123_30_execution10/README.md)
- [五策略汇总、来源和配对统计](artifacts/rollout/formal_seed123_30_execution10/comparison_summary.json)
- [ACT 与 Diffusion 原始 rollout](artifacts/rollout/baselines_task34_real_v2_formal)
- [三种 SmolVLA 原始 rollout](artifacts/rollout/formal_seed123_30_execution10)
- [ACT 与 Diffusion 验证集结果](artifacts/baselines/task34_real_v2)
- [完整实验日志](docs/experiment_log.md)

~~~bash
python -m pip install -e ".[dev,plot]"
python -m pytest -q
~~~

模型权重、数据集、缓存和日志由 [.gitignore](.gitignore) 排除。
