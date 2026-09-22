# T10 朴素异步与 RTC 1.25 配对验证

用户确认沿用 T2 初始状态27–31。两种执行方式，两种延迟（自然、+100ms），原8任务，每条件40回合，共160正式回合。四套件task0/init13各方式各延迟共16预检，不并入正式结果。

只改模型采样为官方T10，实际10次Euler更新，t=1至0.1、dt=-0.1；不训练、不加载学生。RTC权重1.25、EXP、horizon10，控制20Hz、chunk50、执行10个有效动作后重新预测、过期前缀裁剪及所有错误门槛保持原样。RTC实际延迟达到10tick时暂停，不能扩大范围或降低频率。

本轮使用已看过T2结果的状态，属于追加探索。任务、状态哈希、权重文件与T2清单核验一致，调度顺序完全一致。S1原始状态不同，不能用跨状态差异证明蒸馏因果。

## 预检停止

10次积分与RTC数值有效性检查通过；Goal task0/init13的async_d0、rtc_d0、async_d100三回合完成。第四回合rtc_d100在首块接纳处触发 `Initial RTC delay exceeds horizon`，控制tick尚未开始，requests为空，未记录该首块的精确毫秒耗时。正式回合0/160，没有成功率结果。

原代码以 `ceil(elapsed_s * 20) >= 10` 拒绝首块。因此“500ms门槛”只是10tick范围的简写，实际向上取整可在耗时超过450ms时触发。不能根据这条错误宣称实际测得超过500ms，也不能视为运行中观测已过期。未改变该门槛，未重试。完整预检、3回合及错误证据已回传 `artifacts/t10_rtc125_pair_v1`，GPU进程结束，自动检查已删除。

## 用户批准的 v2 首块修正

用户随后明确批准：允许控制时钟启动前的首块等待，不用该等待时长直接拒绝首块；保留实际耗时和向上取整后的延迟估计，不压低、不截断。运行中RTC实际年龄达到10tick仍暂停，其余执行协议不变。

独立入口 `scripts/evaluate_t10_rtc_initial_wait.py`，独立控制器 `scripts/rtc_initial_wait_controller.py`，`InitialWaitTimeline` 只覆盖 prime 方法，继承原运行中 submit/accept/pop。初始请求在 prime 之前记录以保留失败证据。原S1/T2/T10入口未改。v2清单明确记录首块策略变化，不能称为与历史代码完全相同。

本地20项RTC测试通过，包括旧首块拒绝不变、新首块不裁剪、.55s保留11tick估计、运行中10tick拒绝并记录、9tick接纳并裁剪，以及低于旧首块门槛时状态一致。

v2独立输出 `/root/autodl-tmp/outputs/t10_rtc125_initialwait_v2`，日志 `/root/autodl-tmp/logs/t10_rtc125_initialwait_v2.log`，screen `t10-rtc125-v2`。从头重新预检，不混用v1预检；全部16回合通过后才能正式评测。遇到新异常继续暂停，不自动放宽运行中门槛。

### v2 实测停止证据

v2数值预检通过，完成1个异步检查回合；Goal task0/init13的RTC自然延迟检查在控制tick20暂停。首块448.72ms正常接纳（初始实际观测年龄0），随后tick10请求、tick20尝试接纳，实际年龄10tick。请求预计9tick，模型预测444.70ms，端到端455.99ms，10次前向和10次RTC调用，无人为100ms注入。结果在50ms边界接纳，触发原 `RTC delay exceeds horizon`。

这证明当前运行中也会触发门槛，不能仅归因于首块检查，也不能把455.99ms请求耗时说成实测计算500ms。正式结果0/160。记录与视频已备份到 `artifacts/t10_rtc125_initialwait_v2`，进程结束、GPU空闲、自动检查已删除。没有重试或扩大horizon；继续需新的配置决策。

- 入口 `scripts/evaluate_t10_rtc.py`；适配器 `scripts/t10_rtc_policy.py`。
- 远端输出 `/root/autodl-tmp/outputs/t10_rtc125_pair_v1`。
- 日志 `/root/autodl-tmp/logs/t10_rtc125_pair_v1.log`；screen `t10-rtc125`。
- 本地编译、15项已有RTC单元测试及清单一致性断言通过；GPU预检与实时门槛另由远端执行验证。
- 同线程配置、单GPU独占、两进程、batch1，不与其他计算实验并行，不自动开关机。

```sh
cd /root/autodl-tmp/smolvla-flow
PYTHONPATH=src:. HF_HOME=/root/autodl-tmp/huggingface-cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py \
/root/autodl-tmp/venvs/smolvla-flow/bin/python -u -m scripts.evaluate_t10_rtc \
 --output /root/autodl-tmp/outputs/t10_rtc125_pair_v1 \
 --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
 --assets-dir /root/autodl-tmp/libero-assets \
 --source /root/autodl-tmp/outputs/s1_rtc_split_formal_v1 \
 --t2-source /root/autodl-tmp/outputs/t2_rtc125_pair_v1
```
