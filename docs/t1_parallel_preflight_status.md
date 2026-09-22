# T1 并行评测准备状态

2026-09-17。状态：**批量一致性门槛未通过，正式评测暂停，尚未实现/运行完整并行调度器及 400 回合。**

## 实测资源

AutoDL 容器 cpu.max = `2000000 100000`（20 核），memory.max = `96636764160`（90 GiB）。GPU 24564 MiB。检查结束后 GPU 0 MiB、无 screen；实例未自动关机。

## 检查和边界

- 新增 `scripts/probe_libero40_t1_parallel.py`，使用冻结评测入口的模型加载、观测处理和环境初始化。
- 官方权重及 S5 010000.pt 哈希通过，S5 SHA256：`61ae24f990e5cb290e9dedaac5d82daabd6e4e3910449a09b1b13d001f52fe81`。
- 实测 denoise_step 调用次数 T10/T5/S5/T2/T1 为 10/5/5/2/1；时间网格正确，首动作与 select_action 显式噪声路径对齐。
- 保存 Long 10 个任务、初始状态 0/1 稳定后的 20 份观测，每观测 4 份噪声。未执行策略闭环，不产生新的成功率。
- 批量 1/2/4/8/16/32，32 个样本，正序及逆序测试。固定 rtol=1e-5、atol=1e-6，没有调整。
- batch=1 两种顺序完全一致。所有 batch>1 失败，混合样本最大绝对差 0.07858–0.12121（归一化动作）。
- `scripts/check_t1_homogeneous_batch.py` 进一步用完全相同观测和噪声重复拼批，无跨任务或语言长度差异。batch>1 仍失败：

| batch | 完整动作块最大差 | 首动作最大差 |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 0.0451543 | 0.0375516 |
| 4 | 0.0345480 | 0.0319917 |
| 8 | 0.0399772 | 0.0399772 |
| 16 | 0.0378299 | 0.0362455 |
| 32 | 0.0488462 | 0.0405895 |

该反例排除了“仅由不同语言长度和补齐造成”的解释。模型含 241 个 float32 参数张量、547 个 bfloat16 参数张量，保持现有加载精度；尚未定位数值差异具体产生的层，不能将精度确定为唯一原因。独立 CUDA Generator 与原全局随机流连续 8 次采样逐值相同。

## 诊断结果

均为相同观测/噪声下相对 T10 的归一化动作 MSE，80 个观测噪声对；每模型 80 次，共 240 次诊断预测，不含前置门槛调用。

| 模型 | 首动作 MSE | 完整动作块 MSE |
|---|---:|---:|
| T5 | 0.0022505301 | 0.0017714268 |
| S5 | 0.0003706334 | 0.0007282834 |

S5 在这些起点更接近教师。不能由此推出闭环成功率，也没有覆盖整个 rollout 的偏离状态。逐位置、动作分组、逐观测结果在 diagnosis.json；反归一化动作统计尚未交付。

## 视频与旧记录保护

`artifacts/libero40_t1_video_audit_v1/` 保存 32 段旧视频抽帧、选择清单、帧数校验及人工检查边界。旧入口 video 字段会重复指向首个同类视频，不是每回合独立录像。未发现抽帧中明确的成败反转，接触等不可见条件标为不确定。

检查前后本地旧记录 1614 个 JSON 哈希相同。远端旧正式回合、权重和旧入口没有修改。T2 既有视频下载到本地，没有新增 rollout。

## 实际执行

本地三个新增脚本 `python3 -m py_compile` 通过。首轮 probe_v1 在拼批处理转移元数据 None 时失败；修复后在独立 probe_v2 目录运行完成，未覆盖首轮证据。辅助脚本首次遗漏 HF_HOME，离线加载失败，补全相同缓存环境后完成。

远端工作目录 `/root/autodl-tmp/smolvla-flow`，运行环境：

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=4 MUJOCO_GL=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/huggingface-cache
export DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py
/root/autodl-tmp/venvs/smolvla-flow/bin/python scripts/probe_libero40_t1_parallel.py \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --student /root/autodl-tmp/outputs/distill40_s5_b16_v1/stage1/010000.pt \
  --assets-dir /root/autodl-tmp/libero-assets \
  --output /root/autodl-tmp/outputs/libero40_t1_probe_v2
/root/autodl-tmp/venvs/smolvla-flow/bin/python scripts/check_t1_homogeneous_batch.py \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --probe /root/autodl-tmp/outputs/libero40_t1_probe_v2
```

原始输入、噪声、预测和 JSON 已下载至 `artifacts/libero40_t1_probe_v2/`。同名原始运行目录不可覆盖，重新检查须使用新目录。

## 待决定

按已批准停止条件，未进入并发吞吐测试、8 回合执行器检查及 400 回合正式 T1。不能声称 GPU 已满载或批量评测已通过。

建议如用户同意，改为多 CPU 环境准备数据、GPU 保持 batch=1 的供数重叠方案，然后重新检查调度/RNG/闭环一致性。此方案保留原单样本算术路径，但最高吞吐和利用率需另测，不能承诺 90%。不自行放宽容差、不改精度、不以未通过的动态批量继续正式评测。
