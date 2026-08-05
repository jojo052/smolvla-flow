# 异步闭环推理运行时

## 为什么需要异步

20 Hz 控制周期只有 50 ms。2 步学生的 policy-only 推理延迟约 135 到 175 ms，加入动作后处理后单次策略请求仍可能跨越多个控制周期。控制线程如果同步等待模型，动作队列会出现空窗。

运行时拆成两个线程：

```mermaid
flowchart LR
    O[观测] --> C[20 Hz 控制线程]
    C --> Q[动作队列]
    C --> T{剩余动作 <= 10?}
    T -->|是| S[Policy Server 提交最新观测]
    S --> W[推理线程]
    W --> R[RTC: delay + previous prefix + horizon]
    R --> M[新 chunk]
    M --> F[10 步重叠融合]
    F --> Q
    Q --> A[执行 1 个动作]
```

控制线程只做 `tick`、队列消费和安全后处理。模型计算放在 Policy Server 的单 worker 中。待处理请求最多保留一个，新观测到来时替换旧请求，避免策略落后于真实状态。

## 一次闭环 tick

1. 读取当前队列深度。
2. 深度小于等于 10，且推理线程空闲时提交当前观测。
3. 提交请求时保存旧队列前 10 个未执行动作，作为 RTC 的 `prev_chunk_left_over`。
4. 推理线程测量本次耗时。传给下一次 RTC 的 `inference_delay` 为 `ceil(耗时 × 20)`，单位是控制步数。
5. 新 chunk 返回后，先丢弃新 chunk 的前 `inference_delay` 步。这些步对应推理期间已经过去的控制时间。随后旧队列和新 chunk 的前 `L=10` 步做线性融合：

   ```text
   a[i] = (1 - (i + 1) / L) * a_old[i]
          + ((i + 1) / L) * a_new[i]
   ```

   旧队列的过期尾部被丢弃，丢弃 delay 后的新 chunk 第 11 步以后直接接入队列。
6. 控制线程消费一个动作。夹爪迟滞在融合后执行，保证连续动作和夹爪状态分别处理。

## RTC 参数

真实 SmolVLA 调用使用：

```python
policy.predict_action_chunk(
    batch,
    prev_chunk_left_over=previous_prefix,
    inference_delay=measured_delay_steps,
    execution_horizon=10,
)
```

当前配置为 `RTC enabled=true`、指数 prefix schedule、guidance weight 10、execution horizon 10。RTC 负责采样过程中的前缀约束，动作队列负责控制线程和推理线程解耦，普通重叠融合负责 chunk 返回后的最终队列连续性。两层都保留，便于后续做消融。

## 冷启动和安全边界

模型第一次 CUDA 前向包含加载和 kernel warmup，不能直接拿它作为控制线程的实时延迟。实际运行时先同步生成一个初始 chunk，再启动 20 Hz 控制循环。后续推理在线程中执行。

夹爪方向尚未完成真机和仿真确认。代码在硬件模式下拒绝 `gripper_polarity=pending`，当前 smoke 使用 `positive_open` 只代表软件链路测试。队列耗尽时运行时返回 `waiting_for_policy=true`，真机接入时应绑定保持上一安全动作或急停策略，不能把空动作当成合法控制命令。

## 已完成的真实 smoke

产物：`artifacts/async_smolvla_smoke.json`。

队列和 RTC 使用归一化动作，控制线程出队后调用 LeRobot postprocessor，再交给执行端。

| 指标 | 结果 |
| --- | ---: |
| 学生模型 | 2 步 SmolVLA，task 0 |
| 初始 warmup | 498.8 ms |
| 控制 tick | 120 |
| 运行时间 | 6.42 s |
| 输出动作数 | 120 |
| waiting tick | 0 |
| 提前触发次数 | 3 |
| 推理耗时 | 154.2 ms、145.6 ms、206.0 ms |
| RTC delay | 0 → 4 → 3 → 5 步 |
| 合并前队列深度 | 7、7、9 |
| 丢弃新 chunk 前缀 | 4、3、5 步 |
| deadline miss | 0 |
| 输出形状 | `[1,50,7]` |

这次 smoke 验证了异步执行和 RTC 参数传递。它重复使用同一条观测，没有进行 LIBERO 环境成功率评估。

同一运行时接入 50-step 蒸馏得到的 `student_2_action_expert.pt` 后，初始 warmup 为 495.6 ms，120 个 tick 全部产生动作，waiting tick 为 0。两次提前推理耗时为 213.8 ms 和 230.7 ms，RTC delay 为 0、5、5 步，合并前队列深度为 5、5，丢弃新 chunk 前缀为 5、5 步，deadline miss 为 0。该次 smoke 在控制线程出队后调用 LeRobot postprocessor，结果保存于 `artifacts/distillation/task0_dev50/async_smolvla_smoke.json`。该测试仍使用固定观测，不能替代 LIBERO 环境 rollout。

## 文件入口

- `src/smolvla_flow/async_runtime.py`
- `scripts/run_async_runtime_smoke.py`
- `scripts/run_async_smolvla_smoke.py`
- `tests/test_async_runtime.py`
