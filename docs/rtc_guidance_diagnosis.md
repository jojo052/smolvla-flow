# S1 / T2 RTC 引导强度诊断（2026-09-19）

本次仅做隔离推理诊断及条件触发的小规模闭环验证。不训练、不覆盖原 240 回合，不将本次预检状态上的结果作为正式泛化结果。

## 设计与来源

- 四套件 task 0，官方 init 13。重放原 async / 自然延迟预检记录的动作，取控制 tick 10、30、60 的观测，共 12 个。
- 每观测两份显式噪声，预计延迟 3 / 5 tick；每模型每权重 48 组。
- 权重固定为 0、0.1、0.5、1、2、10，共 576 次当前观测预测，另有前缀生成与无引导参考预测。
- 共同前缀来自 S1 在十个 tick 前观测上的无引导预测，去掉前十个动作。所有模型与权重使用相同观测、当前噪声和前缀。对 T2，这是共享 S1 前缀的机制诊断，不能替代 T2 自身闭环。
- 逐次检查积分次数、时间点、有限性；S1 一次、T2 两次。基础权重和处理器的 8 个文件 SHA-256 均与旧 manifest 相符，S1 哈希和锁定代码亦核对。
- 保存观测、噪声、前缀、输出数组及哈希，路径 `artifacts/rtc_guidance_diagnosis_v1/`。

正式运行前规定：只在权重不超过 1 的候选中选取所有样本加权前缀误差不增大、且全块最大归一化绝对幅值不超过原预测/旧前缀幅值上界的最大权重。容差分别为 1e-6 和 1e-5。该门控检查过度修正，不证明任务能力。

## 固定输入结果

下表为加权前缀 MSE 之和除以各模型自身无引导 MSE 之和。包括预计过期的前缀位置；不是成功率，也不是对专家动作的误差。

| 权重 | S1 相对误差 | T2 相对误差 |
|---|---:|---:|
| 0 | 1.000 | 1.000 |
| 0.1 | 0.829 | 0.827 |
| 0.5 | 0.316 | 0.401 |
| 1 | 0.058 | 0.166 |
| 2 | 0.820 | 0.028 |
| 10 | 68.220 | 0.239 |

S1 权重 1：48/48 通过。权重 10：48/48 的误差增大，48/48 超出幅值上界，最大归一化幅值为 36.03。S1 权重 2 虽全部改善前缀 MSE，但 29/48 超出幅值上界。

T2 权重 10 的平均误差下降，但 1/48 样本误差增大，24/48 超出幅值上界，最大超出量 3.11。T2 没有表现出 S1 同等程度的放大，仍不能据此宣布其为有效 RTC 教师。门控专门用于选择 S1 低权重候选，不将 T2 未全通过直接解释成闭环失败。

## 条件触发的闭环检查

按预设规则选择 S1 权重 1。使用独立入口 `scripts/check_rtc_low_guidance.py`，四套件 task0 / init13，朴素异步与 RTC，延迟 0 / 100ms，共 16 回合。随机种子 123 打乱条件顺序，保持原 split-process / batch1、20Hz、动作上限、实际过期裁剪和 5ms / 5% 门控。原执行器与历史参数默认值不修改。

独立输出为 `/root/autodl-tmp/outputs/rtc_low_guidance_check_v1`。不根据闭环成功率追加权重搜索，出现异常停止。结果以该目录的 `summary.json` 和逐回合记录为准。

16 回合已全部完成，最大单回合迟到比例 0.7874%，低于原 5% 门槛；逐回合端到端时间合计 269.91 秒。原正式目录仍有 240 条记录，没有向其中写入本轮回合。

| 条件 | 朴素异步 | RTC 权重 1 |
|---|---:|---:|
| 自然延迟 | 2/4 | 2/4 |
| +100ms | 2/4 | 3/4 |

自然延迟两组均为 Goal、Object 成功，Spatial、Long 失败。+100ms 的 RTC 多成功一个 Long 回合。每条件仅四任务各一回合，不能据此宣布 RTC 成功率显著提高。

| 条件 | 平移指令块边界跳变均值 async / RTC | 旋转跳变 async / RTC | 夹爪跳变 async / RTC |
|---|---:|---:|---:|
| 自然延迟 | 0.3065 / 0.1371 | 0.0486 / 0.0207 | 0.2513 / 0.1269 |
| +100ms | 0.3214 / 0.1493 | 0.0485 / 0.0258 | 0.2755 / 0.1526 |

上述为各自实际轨迹的描述性指标，非固定状态上的因果效应，也非物理位移单位。+100ms 的运动指令超 1 tick 比例为 async 0.199%、RTC 0.529%，所以不能宣称低权重改善了所有幅值指标。自然延迟对应为 0.615% / 0.298%。

本轮支持：S1 权重 10 的过度修正有直接机制证据；权重 1 消除了固定输入测试中的误差放大，并在小规模闭环中减少了块边界指令跳变，没有重现此前的严重成功率崩溃。尚未证明稳定成功率增益，也未证明 T2+RTC 是合格教师。结束后不追加蒸馏或扩大正式评估。

## 验证命令

```sh
python3 -m py_compile scripts/diagnose_rtc_guidance.py scripts/check_rtc_low_guidance.py
PYTHONPATH=src:. python3 -m unittest discover -s tests -p 'test_rtc*.py' -q
```

编译检查通过；已有 RTC 记账/RPC 单元测试 15 项通过。GPU 行为另外由隔离运行检查，不能用这 15 项测试替代。

服务器运行命令（须使用新的输出目录复做，不能覆盖本轮结果）：

```sh
cd /root/autodl-tmp/smolvla-flow
export PYTHONPATH=src:.
export HF_HOME=/root/autodl-tmp/huggingface-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py
/root/autodl-tmp/venvs/smolvla-flow/bin/python -u -m scripts.diagnose_rtc_guidance \
  --output /root/autodl-tmp/outputs/rtc_guidance_diagnosis_v1 \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --student /root/autodl-tmp/outputs/distill40_s1_formal_v1/training/010000.pt \
  --assets-dir /root/autodl-tmp/libero-assets \
  --source /root/autodl-tmp/outputs/s1_rtc_split_formal_v1
/root/autodl-tmp/venvs/smolvla-flow/bin/python -u -m scripts.check_rtc_low_guidance \
  --output /root/autodl-tmp/outputs/rtc_low_guidance_check_v1 \
  --diagnosis /root/autodl-tmp/outputs/rtc_guidance_diagnosis_v1 \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --student /root/autodl-tmp/outputs/distill40_s1_formal_v1/training/010000.pt \
  --assets-dir /root/autodl-tmp/libero-assets \
  --source /root/autodl-tmp/outputs/s1_rtc_split_formal_v1
```
