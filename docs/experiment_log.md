# 实验记录

这个文件记录已经运行过的实验，参数和结果优先从对应 JSON 产物读取。新的实验追加新条目，不覆盖旧结果。

## 2026-08-03 至 2026-08-04：LIBERO-Spatial task 0 教师预检

### 运行目的

确认官方 LIBERO SmolVLA checkpoint 能否在当前 LIBERO 数据接口上完成真实样本前向，并测量 10、5、2 个 Flow Matching 采样步的推理延迟。5 步和 2 步结果只代表同一个教师减少积分步数的延迟参考，不能当作蒸馏学生质量结果。

### 环境与数据

| 字段 | 记录 |
| --- | --- |
| 计算路径 | Mac Codex → SSH 2222 → WSL Ubuntu → RTX 4090 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| LeRobot | 0.6.1 |
| 数据集 | `HuggingFaceVLA/libero` |
| suite | `libero_spatial` |
| suite task | 0 |
| dataset task index | 34 |
| 语言指令 | `pick up the black bowl between the plate and the ramekin and place it on the plate` |
| task episode 数 | 45 |
| task frame 数 | 4487 |
| 真实前向 episode | 1272 |
| 输入状态 | 8 维 |
| 输入图像 | 2 路，`3×256×256` |
| 输出动作 | 7 维，chunk 长度 50 |
| 夹爪维度 | zero-based index 6，也就是第 7 维 |

夹爪动作在 task 0 数据中覆盖 `[-1, 1]`。均值为 `-0.0363272`，标准差为 `0.999340`，因此 ±0.5 迟滞阈值两侧都有数据。开合方向仍需通过仿真动作效果确认。

### 教师 checkpoint

| 字段 | 结果 |
| --- | --- |
| checkpoint | `HuggingFaceVLA/smolvla_libero` |
| Flow steps | 10 |
| strict state dict load | 通过，missing 和 unexpected keys 均为 0 |
| 基础 VLM 权重 | 设置 `load_vlm_weights=false`，只加载配置和 tokenizer 文件，policy 权重覆盖完整参数 |
| 模型输出 | `[1, 50, 7]` |
| 输出 finite | true |
| 模型加载时间 | 9.5230 s |
| 首次前向时间 | 0.5199 s |
| 峰值显存 | 1,581,556,736 bytes |

### 延迟测量

测试方式为 5 次 warmup，随后 10 次测量，CUDA 事件同步后统计。样本为 episode 1272 的 frame 0。结果保存于 `artifacts/preflight/teacher_forward_task0.json`。

| Flow steps | 模型状态 | 平均 | 中位数 | P95 | 20 Hz 控制周期数 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | 原生教师 | 428.68 ms | 452.18 ms | 562.55 ms | 8.57 |
| 5 | 未蒸馏教师少步采样 | 226.83 ms | 197.76 ms | 341.88 ms | 4.54 |
| 2 | 未蒸馏教师少步采样 | 162.91 ms | 167.35 ms | 220.64 ms | 3.26 |

这些数值说明 20 Hz 控制线程不能等待模型返回。动作队列需要在控制线程和推理线程之间解耦，RTC 需要使用真实 inference delay 做重叠前缀约束。2 步少采样的延迟下降不能替代 10 到 5 到 2 的质量蒸馏。

### RTC smoke

当前 LeRobot 的 RTC 模块路径为 `lerobot.policies.rtc`。修复导入后完成以下检查：

- `RTCProcessor` 成功构造。
- `denoise_step` 使用 10 步 execution horizon、4 步 inference delay 完成一次 guidance。
- 输出形状为 `[1, 50, 7]`，全部 finite。
- `ActionQueue` 合并 50 步动作并消费 1 步后，剩余 45 步。
- 配置为 `enabled=true`、`execution_horizon=10`、`max_guidance_weight=10.0`、`prefix_attention_schedule=EXP`。

结果保存于 `artifacts/preflight/rtc_smoke_fixed.json`。

### 尚未完成

- ACT 100 step 冒烟训练和 LIBERO episode。
- Diffusion Policy 100 step 冒烟训练和 LIBERO episode。
- 异步队列在实际环境中的 deadline miss、等待时间和轨迹抖动测量。
- 夹爪正负方向确认。
- 10 Hz 数据动作到 20 Hz 控制周期的插值策略确认。
- 正式多 seed 评测和 LIBERO 成功率。

## 2026-08-04：动作专家 10 到 5 到 2 蒸馏开发实验

### 运行目的

把 SmolVLA 原生 10 步 Flow Matching action expert 逐阶段压缩到 5 步和 2 步。每个阶段都使用冻结教师轨迹一致性损失和 7D 动作回归损失，学生从上一阶段权重初始化。

### 运行配置

| 字段 | 记录 |
| --- | --- |
| checkpoint | `HuggingFaceVLA/smolvla_libero` |
| 数据 | LIBERO-Spatial task 0，episode 1272 的 8 个完整动作 chunk |
| 每阶段更新 | 5 steps |
| chunk | 50 |
| 模型内部动作维度 | 32，前 7 维对应真实动作 |
| 蒸馏损失动作维度 | 7，padding 维度不参与 loss |
| 时间方向 | `t=1` 到 `t=0` |
| 动作空间 | LeRobot `MEAN_STD` 归一化空间 |
| 训练参数 | 97,420,192 个 action expert 参数，VLM 和 state projection 冻结 |
| 语义损失 | 关闭 |
| 尾 chunk | 只保留 50 帧完整 chunk，跳过不足 50 帧的伪目标 |

### 阶段结果

| 阶段 | 用时 | 平均每步 | 峰值显存 | 首个 loss | 最后 loss | 最后轨迹 loss | 最后动作 loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 → 5 | 5.42 s | 1.084 s | 3.83 GiB | 0.330307 | 0.265619 | 0.005804 | 0.259815 |
| 5 → 2 | 3.52 s | 0.705 s | 3.95 GiB | 0.318390 | 0.322103 | 0.016297 | 0.305806 |

两阶段均完成 finite 检查和参数更新。两个阶段的 `action_in_proj.weight` 首步平均参数变化分别为 `9.9979e-06` 和 `9.9980e-06`。5 → 2 只有 5 个更新步，当前结果用于验证训练链路，不能推断任务成功率。

### 蒸馏后延迟

在 episode 1272 frame 0，5 次 warmup、10 次测量：

| 模型 | Flow steps | 平均 | 中位数 | P95 | 输出 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原生教师 | 10 | 418.01 ms | 420.07 ms | 491.82 ms | `[1,50,7]` |
| 蒸馏学生 | 2 | 134.78 ms | 130.46 ms | 159.96 ms | `[1,50,7]` |

相对这次同一批次测得的 10 步教师，2 步学生平均延迟约为 32.2%，约 3.10 倍加速。相同随机噪声下的动作 MSE 为 0.08155，MAE 为 0.19573。这个动作差异只用于 smoke 记录，质量评估需要固定验证集和 LIBERO rollout。

### 产物

- `artifacts/distillation/task0_dev5_final/distillation_metrics.json`
- `artifacts/distillation/task0_dev5_final/benchmark.json`
- 远程 4090 同目录保存 `student_5_action_expert.pt` 和 `student_2_action_expert.pt`，每个约 194 MB，只包含可训练 action expert 参数。

### 下一步入口

1. 先扩大 task 0 的蒸馏样本和更新步数，保存固定验证 chunk 的教师学生动作误差。
2. 固定 task 0 的 episode 划分，运行 ACT 与 Diffusion Policy 的 100 step 冒烟链路。
3. 在 LIBERO 环境中做原生教师、蒸馏学生和未蒸馏少步模型的 rollout 对比。
4. 把 20 Hz 控制周期、10 步重叠、夹爪迟滞、RTC 和动作插值写成统一运行时接口。

## 2026-08-04：异步闭环推理 smoke

### 运行链路

已实现控制线程、Policy Server、动作队列、提前触发、10 步重叠融合、推理延迟测量和 RTC 参数传递。控制频率设为 20 Hz，队列剩余 10 步时提前提交最新观测。新 chunk 返回后先丢弃推理期间对应的 delay 步，再做重叠融合。首次控制前先做一次 CUDA warmup，避免冷启动占用控制周期。

### 真实 2 步学生结果

使用 `student_2_action_expert.pt`，在 episode 1272 的固定观测上重复运行 120 个控制 tick：

队列和 RTC 保留归一化动作，控制线程出队后调用 LeRobot postprocessor，再交给执行端。

| 指标 | 结果 |
| --- | ---: |
| 初始 warmup | 498.8 ms |
| 控制 tick | 120 |
| 输出动作数 | 120 |
| waiting tick | 0 |
| 提前触发 | 3 次 |
| 推理耗时 | 154.2 ms、145.6 ms、206.0 ms |
| RTC delay | 0 → 4 → 3 → 5 步 |
| 合并前队列深度 | 7、7、9 |
| 丢弃新 chunk 前缀 | 4、3、5 步 |
| deadline miss | 0 |
| 输出 | `[1,50,7]`，finite |

结果保存于 `artifacts/async_smolvla_smoke.json`。本次使用固定观测验证线程和队列行为，尚未进入 LIBERO 环境执行真实任务。

### 运行时安全约束

夹爪开合方向仍待确认。硬件模式下 `gripper_polarity=pending` 会拒绝启动，当前 smoke 的 `positive_open` 只用于软件测试。队列耗尽时会报告 `waiting_for_policy`，真机适配时需要接入保持上一安全动作或急停策略。

## 2026-08-04：扩大动作蒸馏更新步数

### 运行配置

在同一个 LIBERO-Spatial task 0、episode 1272 上继续运行，使用该 episode 中所有 9 个完整的 50 帧动作 chunk。每个阶段更新 50 次，仍然只训练 action expert 的 97,420,192 个参数，VLM 和 state projection 保持冻结。

### 阶段结果

| 阶段 | 完整 chunk 数 | 更新步数 | 峰值显存 | 首个 loss | 最后 loss | 最后轨迹 loss | 最后动作 loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 → 5 | 9 | 50 | 3.84 GiB | 0.330307 | 0.152632 | 0.022009 | 0.130623 |
| 5 → 2 | 9 | 50 | 3.95 GiB | 0.081207 | 0.083457 | 0.013091 | 0.070366 |

10 → 5 阶段的最后一步 loss 为 0.152632。5 → 2 阶段从更好的 5 步学生初始化，最后一步动作回归损失为 0.070366。完整逐步记录见 `artifacts/distillation/task0_dev50/distillation_metrics.json`。

### 长训 checkpoint 延迟

使用相同观测、5 次 warmup 和 10 次测量，统计的是 `predict_action_chunk` policy-only 延迟，不包含出队后的 postprocessor：

| 模型 | Flow steps | 平均 | 中位数 | P95 | 输出 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原生教师 | 10 | 408.11 ms | 400.20 ms | 461.87 ms | `[1,50,7]` |
| 50-step 蒸馏学生 | 2 | 133.51 ms | 126.52 ms | 180.76 ms | `[1,50,7]` |

长训 student 的平均推理时间约为教师的 32.7%，同一随机种子下动作 MSE 为 0.111979，MAE 为 0.236593。这个 MSE 比 5-step 开发实验高，说明单个固定观测上的动作差异会受训练样本、随机噪声和当前权重影响，仍需固定验证集和多 seed 评估。

### 长训 student 接入异步闭环

继续使用 20 Hz 控制、chunk 50、执行 10 步、重叠 10 步和 RTC。固定 episode 1272 的观测运行 120 个 tick：

| 指标 | 结果 |
| --- | ---: |
| 初始 warmup | 495.6 ms |
| 输出动作数 | 120 |
| waiting tick | 0 |
| 提前触发 | 2 次 |
| 推理耗时 | 213.8 ms、230.7 ms |
| RTC delay | 0 → 5 → 5 步 |
| 合并前队列深度 | 5、5 |
| 丢弃新 chunk 前缀 | 5、5 步 |
| deadline miss | 0 |

这次结果包含控制线程出队后的 LeRobot postprocessor，只验证固定观测下的运行时调度，尚未等同于 LIBERO 任务成功率。产物为 `artifacts/distillation/task0_dev50/benchmark.json`、`artifacts/distillation/task0_dev50/async_smolvla_smoke.json`。

## 2026-08-04：LIBERO rollout 入口与环境预检

### Rollout 定义

rollout 是一次完整的策略与环境交互过程。环境 reset 后返回多视角图像、机器人状态和任务语言，策略生成一个动作 chunk，环境按控制周期执行动作并返回下一帧观测、reward 和成功标志，直到任务成功、失败或达到最大步数。评估器保留 LeRobot 的 LIBERO 图像翻转、状态拼接、策略预处理和动作反归一化顺序。

### 已实现的评估入口

`scripts/run_libero_rollout.py` 支持以下组合：

| 模式 | 模型 | 用途 |
| --- | --- | --- |
| `sync` | 10 步教师 | 质量和成功率基线 |
| `sync` | 未蒸馏 2 步教师 | 低步采样延迟参考 |
| `sync` | 2 步蒸馏学生 | 蒸馏质量对照 |
| `async` | 2 步蒸馏学生 | 20 Hz、动作队列、RTC 和重叠融合评估 |

每个 episode 保存成功、reward、步数、实际控制频率、等待 tick、推理延迟、动作平滑度和异步 server 事件。

### 4090 预检结果

运行命令：

```bash
python scripts/run_libero_rollout.py --mode sync --flow-steps 10 --episodes 1 --max-steps 1 \
  --output artifacts/rollout/libero_spatial_task0_rollout.json
```

评估器正确进入了 LIBERO 配置和资产检查阶段，随后停止。当前环境缺少官方资产目录：

```text
${LIBERO_ASSETS_DIR}/scenes
${LIBERO_ASSETS_DIR}/articulated_objects
${LIBERO_ASSETS_DIR}/stable_scanned_objects
${LIBERO_ASSETS_DIR}/turbosquid_objects
```

运行环境当前无法访问 Hugging Face，资产下载需要在可联网机器完成，再复制到 `${LIBERO_ASSETS_DIR}`，或者通过 `--assets-dir` 指定复制后的目录。预检记录保存在 `artifacts/rollout/libero_spatial_task0_rollout.json`，状态为 `blocked`。在资产补齐前不报告 LIBERO 成功率。

## 2026-08-05：LIBERO 资产补齐与首轮真实回放

### 资产同步

官方 Hugging Face 下载受网络连接影响未完成。通过 ModelScope 获取完整
`lerobot/libero-assets`，再从可联网机器的 `${ASSET_SOURCE_DIR}`
同步到 GPU 运行环境：

```text
${LIBERO_ASSETS_DIR}
```

四个必需目录均已存在，文件数为 587，目录占用约 405 MB。原先外置盘中的不完整
`${INCOMPLETE_ASSET_SOURCE_DIR}` 保留，没有参与覆盖或删除。

### 预检修复

单环境 `LiberoEnv` 返回的嵌套 `robot_state` 叶节点没有 batch 维度，
`LiberoProcessorStep` 读取四元数时要求 `(B, 4)`。`run_libero_rollout.py`
现在只为嵌套 robot state 叶节点补 batch 维度，图像和顶层策略张量保持原有处理。
本地测试结果为 37 passed。

### 单步预检

10 步教师完成 1 个环境步，输出有限，单步推理耗时 852.2 ms。该记录只用于确认
资产、环境、模型和动作后处理链路，成功标志为 false 属于预期。记录为
`artifacts/rollout/libero_spatial_task0_preflight.json`。

### task 0 回放结果

| 配置 | 步数 | 成功 | 总耗时 | 平均推理 | 实际控制频率 | 动作平滑度均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 步教师，同步 | 81 | 1/1 | 39.12 s | 360.1 ms | 2.07 Hz | 0.0907 |
| 2 步蒸馏学生，异步 RTC | 280 | 0/1 | 25.77 s | 308.9 ms | 10.86 Hz | 0.0360 |

异步学生回放提交 7 次策略请求，7 次完成，没有 deadline miss 或等待 tick。
本轮使用 `student_2_action_expert.pt`，结果记录为
`artifacts/rollout/libero_spatial_task0_student_async.json`；教师记录为
`artifacts/rollout/libero_spatial_task0_teacher_sync.json`。

当前结果表明资产和真实 LIBERO 交互链路已经打通。2 步蒸馏权重在这个 task 0、seed 0
上尚未复现教师成功，需要扩大蒸馏数据和验证 episode 后再判断蒸馏质量。

## 2026-08-05：固定 seed 诊断与 RTC 队列语义修复

### 蒸馏数据量与动作误差

当前 `distillation_metrics.json` 只有 9 个完整动作 chunk。10 → 5 和 5 → 2
两个阶段各更新 50 次，同一随机种子下记录的学生动作 MSE 为 0.11198。这个训练规模
只能作为开发诊断，暂不足以判断蒸馏权重是否失效。

### 固定 seed=123 的 task 0 对照

在相同学生权重、固定 torch seed=123 和 5 个 episode 下完成同步与异步对照：

| 配置 | 成功 | 实际控制频率 | 动作平滑度均值 |
| --- | ---: | ---: | ---: |
| 2 步蒸馏学生，同步 | 3/5 | 4.26 Hz | 0.0744 |
| 2 步蒸馏学生，无 RTC 异步 | 2/5 | 9.86 Hz | 0.03149 |
| 2 步蒸馏学生，RTC 队列 replace 修复后异步 | 2/5 | 9.84 Hz | 0.03607 |

RTC 队列修复遵循 LeRobot 的 original queue 与 execution queue 分离语义。RTC 开启时
按推理延迟丢弃新 chunk 前缀后直接替换，关闭 RTC 时继续使用已有 overlap blend。
修复通过 41 个本地测试。

### 诊断结论

首轮 0/1 结果不能单独说明蒸馏权重失效。当前应将模型质量不足与异步控制降级分开处理：
前者需要增加完整 chunk、更新步数和验证 episode，后者需要继续比较队列替换、RTC 引导、
控制频率和动作平滑度。
