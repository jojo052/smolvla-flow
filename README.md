# SmolVLA Flow-Matching 动作专家蒸馏与异步闭环优化

这个仓库按可验证的小步骤实现 SmolVLA 动作专家蒸馏和异步闭环。第一阶段用二维条件数据验证 Flow Matching 的训练与采样，再把二维点替换为机器人动作 chunk，把 MLP 替换成 SmolVLA 的 Transformer Action Expert。

## 当前状态

| 项目 | 当前结果 | 说明 |
| --- | --- | --- |
| task0 语义映射 | 已确认 | LIBERO-Spatial task 0 对应数据集 `task_index=34` |
| task0 固定划分 | 已实现脚本 | 45 个 episode 可生成 31/7/7 train/validation/test，完整窗口数按 `sample_stride=4` 统计 |
| task0 蒸馏数据 | 开发诊断 | 当前记录只有 9 个完整的 50 帧 chunk |
| 10 → 5 → 2 蒸馏 | 已完成开发链路 | 每阶段 50 次更新的结果只用于诊断，不能代表正式质量 |
| 固定 seed=123 同步对照 | 历史 3/5 | 2 步蒸馏学生，5 个 episode；需用当前 `--torch-seed` 重新生成 |
| 固定 seed=123 异步对照 | 历史 2/5 | 无 RTC 和 RTC replace 两种队列实验均为 2/5；需用当前 `--torch-seed` 重新生成 |
| ACT | 未完成 | 100 step 冒烟和 LIBERO episode 尚未运行 |
| Diffusion Policy | 未完成 | 100 step 冒烟和 LIBERO episode 尚未运行 |
| 多 seed 评测 | 未完成 | 还没有固定验证集上的完整多 seed 结果 |
| 正式评测 | 未完成 | 尚未形成可报告的 LIBERO 成功率和统计区间 |

已有结果表明运行时链路可以执行完整的 LIBERO 交互，但当前蒸馏数据量和评测覆盖范围不足以判断模型质量。实验记录见 [docs/experiment_log.md](docs/experiment_log.md)。

## TODO

1. 在 GPU 环境运行固定划分脚本，确认完整 parquet 数据与 31/7/7 episode manifest 一致。
2. 增加蒸馏更新步数，固定验证 chunk 和随机种子，报告动作 MSE、MAE、轨迹误差和有限值检查。
3. 完成 ACT 与 Diffusion Policy 的 100 step 冒烟训练，再接入相同的 LIBERO episode 划分。
4. 运行原生 10 步教师、未蒸馏少步教师和蒸馏学生的多 seed 同步对照。
5. 运行异步队列、RTC、控制频率和动作平滑度的多 seed 对照，单独记录 deadline miss 和 waiting tick。
6. 确认夹爪正负方向、动作插值和安全保持动作，再开始正式 LIBERO 评测。

## 可复现实验

命令只使用环境变量定位数据、checkpoint 和资产，不把某台机器的用户名、内网地址或绝对路径写入仓库。

### 1. 环境检查

```bash
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export PYTHONPATH="$PROJECT_ROOT/src"

cp .env.example .env
python -m pip install -e ".[dev,plot]"
python scripts/validate_experiment_config.py
pytest -q
```

真实 SmolVLA 实验需要本地可读取的 LeRobot、SmolVLA、LIBERO 和 MuJoCo 依赖。需要离线运行时，将 `HF_HOME` 指向已经准备好的 Hugging Face 缓存，并设置 `HF_HUB_OFFLINE=1`。

### 已验证环境版本

下面的版本来自当前 GPU 预检记录，不保证其他机器完全一致：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| LeRobot | 0.6.1 |
| 额外依赖 | SmolVLA、LIBERO、MuJoCo |

预检结果和限制见 [docs/experiment_log.md](docs/experiment_log.md)。

### 2. task0 数据流

先从 LIBERO 元数据解析 suite task 和数据集 task index，再选择属于 `task_index=34` 的 episode。蒸馏脚本会读取 parquet 行，校验同一 episode 的 task index 一致性，过滤目标 task，按 frame 排序，并跳过不足 50 帧的尾部 chunk。

```bash
export LIBERO_METADATA_ROOT="${LIBERO_METADATA_ROOT:?Set the local dataset root containing meta/}"
export LIBERO_PARQUET="${LIBERO_PARQUET:?Set a local parquet shard}"
export EPISODE_INDEX="${EPISODE_INDEX:?Set an episode resolved from task_index=34 metadata}"

python "$PROJECT_ROOT/scripts/inspect_libero_task.py" \
  --suite libero_spatial \
  --suite-task-id 0 \
  --metadata-root "$LIBERO_METADATA_ROOT" \
  --output "$PROJECT_ROOT/artifacts/preflight/libero_spatial_task0.json"
```

生成固定 episode 划分和完整动作窗口统计。脚本只读取 `meta/`，不会读取或复制图像、动作 parquet 数据：

```bash
python "$PROJECT_ROOT/scripts/build_task_index_data.py" \
  --metadata-root "$LIBERO_METADATA_ROOT" \
  --task-index 34 \
  --chunk-size 50 \
  --sample-stride 4 \
  --seed 0 \
  --output "$PROJECT_ROOT/artifacts/preflight/libero_spatial_task0_split.json"
```

当前 45 个 task34 episode 的元数据统计为 4487 帧、584 个 stride=4 完整窗口、61 个不重叠完整窗口。实际训练前仍需确认对应数据 shard 全部可读。

下面的命令展示 task0 的语义过滤流程。`EPISODE_INDEX` 应来自元数据，不再把单个历史 episode 写死在命令里。

```bash
python "$PROJECT_ROOT/scripts/distill_smolvla_action_expert.py" \
  --parquet "$LIBERO_PARQUET" \
  --task-index 34 \
  --episode-index "$EPISODE_INDEX" \
  --sample-stride 4 \
  --max-samples 0 \
  --steps-per-stage 50 \
  --learning-rate 1e-5 \
  --seed 0 \
  --output-dir "$PROJECT_ROOT/artifacts/distillation/task0_task34"
```

输出包含 `task_index_filter`、筛选前后的样本数、两个蒸馏阶段的逐步 loss、finite 状态和可训练参数信息。当前已记录的 task0 开发实验只覆盖 9 个完整 chunk。

### 3. 延迟和动作差异

选择一个已经通过 task0 过滤的 episode，使用蒸馏得到的 action expert 权重：

```bash
export ADAPTER_PATH="${ADAPTER_PATH:?Set student_2_action_expert.pt}"

python "$PROJECT_ROOT/scripts/benchmark_distilled_smolvla.py" \
  --adapter "$ADAPTER_PATH" \
  --parquet "$LIBERO_PARQUET" \
  --episode-index "$EPISODE_INDEX" \
  --warmup 5 \
  --repeats 10 \
  --seed 0 \
  --output "$PROJECT_ROOT/artifacts/distillation/task0_task34/benchmark.json"
```

### 4. 异步运行时和 LIBERO

先运行不依赖模型的队列 smoke：

```bash
PYTHONPATH="$PROJECT_ROOT/src" \
  python "$PROJECT_ROOT/scripts/run_async_runtime_smoke.py" \
  --ticks 70 \
  --latency-seconds 0.08
```

真实学生的 RTC smoke：

```bash
python "$PROJECT_ROOT/scripts/run_async_smolvla_smoke.py" \
  --adapter "$ADAPTER_PATH" \
  --parquet "$LIBERO_PARQUET" \
  --episode-index "$EPISODE_INDEX" \
  --ticks 120 \
  --output "$PROJECT_ROOT/artifacts/async_smolvla_smoke.json"
```

LIBERO rollout 需要本地资产目录。通过 `--assets-dir` 指定包含 `scenes`、`articulated_objects`、`stable_scanned_objects` 和 `turbosquid_objects` 的目录：

```bash
export LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:?Set the local libero-assets directory}"

python "$PROJECT_ROOT/scripts/run_libero_rollout.py" \
  --mode sync \
  --flow-steps 10 \
  --episodes 1 \
  --assets-dir "$LIBERO_ASSETS_DIR" \
  --output "$PROJECT_ROOT/artifacts/rollout/teacher_task0.json"

python "$PROJECT_ROOT/scripts/run_libero_rollout.py" \
  --mode async \
  --flow-steps 2 \
  --adapter "$ADAPTER_PATH" \
  --episodes 1 \
  --assets-dir "$LIBERO_ASSETS_DIR" \
  --output "$PROJECT_ROOT/artifacts/rollout/student_task0_async.json"
```

输出会记录成功率、累计 reward、实际控制频率、等待 tick、推理延迟、动作平滑度和异步队列事件。

固定 seed 对照需要让策略采样噪声可复现。`run_libero_rollout.py` 的 `--torch-seed`
在每个 episode 的首次策略前向处重置 Python、NumPy 和 PyTorch（含 CUDA）RNG，
并把同一值传给推理线程，worker 线程在自己的线程里重新播种（PyTorch 默认 CPU
generator 是线程局部的）。同步与异步使用相同起始值，后续噪声序列仍可能受 warmup 和线程调度影响。
不传该参数时保持原有的非种子行为。

```bash
python "$PROJECT_ROOT/scripts/run_libero_rollout.py" \
  --mode sync \
  --flow-steps 2 \
  --adapter "$ADAPTER_PATH" \
  --episodes 5 \
  --start-seed 0 \
  --torch-seed 123 \
  --assets-dir "$LIBERO_ASSETS_DIR" \
  --output "$PROJECT_ROOT/artifacts/rollout/student_task0_sync_fixed.json"

python "$PROJECT_ROOT/scripts/run_libero_rollout.py" \
  --mode async \
  --flow-steps 2 \
  --adapter "$ADAPTER_PATH" \
  --episodes 5 \
  --start-seed 0 \
  --torch-seed 123 \
  --assets-dir "$LIBERO_ASSETS_DIR" \
  --output "$PROJECT_ROOT/artifacts/rollout/student_task0_async_fixed.json"
```

## 失败记录与实验日志

- [完整实验日志](docs/experiment_log.md)：包含教师预检、蒸馏开发、异步 smoke、固定 seed 诊断和 LIBERO 回放。
- [异步运行时说明](docs/async_runtime.md)：包含队列、重叠融合、RTC 和控制线程约束。
- [task34 固定划分 manifest](artifacts/preflight/libero_spatial_task0_split.json)：记录 31/7/7 episode 划分和完整动作窗口统计。
- [LIBERO 资产缺失预检](artifacts/rollout/libero_spatial_task0_rollout.json)：记录为 `blocked`，不应当解释为成功率为零。
- [task0 教师同步回放](artifacts/rollout/libero_spatial_task0_teacher_sync.json)：首轮真实回放为 1/1。
- [task0 学生异步回放](artifacts/rollout/libero_spatial_task0_student_async.json)：首轮真实回放为 0/1，需要扩大数据和验证 episode。
- [seed=123 同步结果](artifacts/rollout/remote_seed123/student_sync_5episodes_seed123.json)：固定学生权重下为 3/5。
- [seed=123 异步结果](artifacts/rollout/remote_seed123/student_async_rtc_official_queue_5episodes_seed123.json)：RTC replace 对照为 2/5。

小型 JSON 和 JSONL 记录保留在 `artifacts/`，权重、缓存和大数据由 `.gitignore` 排除。

## 方法与实施顺序

1. 二维条件 Flow Matching。
2. Transformer Action Expert 和动作 chunk。
3. 10 步教师模型。
4. 10 → 5 → 2 步渐进蒸馏。
5. 异步动作队列、提前触发和重叠 chunk 融合。
6. SmolVLA 与 LIBERO 接入。
7. ACT、Diffusion Policy、原生 SmolVLA 和蒸馏模型对照实验。

`src/smolvla_flow/action_expert.py` 将 Flow Matching 扩展到形状为 `[batch, chunk_size, action_dim]` 的连续动作。每个动作位置通过自注意力交换轨迹信息，再通过交叉注意力读取上下文 token。真实 SmolVLA 路径会使用图像、语言和机器人状态编码。

## 第一阶段：二维条件 Flow Matching

模型接收带噪二维点 `x_t`、连续时间 `t` 和条件，预测从噪声 `x_0` 流向数据 `x_1` 的速度：

```text
x_t = (1 - t) * x_0 + t * x_1
目标速度 = x_1 - x_0
```

推理从高斯噪声出发，用欧拉法积分预测速度。`--sample-steps 10` 对应原生多步采样，之后再比较 5 步和 2 步模型。

## 异步闭环约束

当前运行时配置固定为 20 Hz 控制、chunk 50、执行 10 步、重叠 10 步和 RTC。控制线程与推理线程解耦，队列保留归一化动作，控制线程出队后再调用 LeRobot postprocessor。夹爪方向和硬件安全保持动作确认前，硬件模式会拒绝启动。
