# 官方T5：朴素异步与RTC1.25

用户在T10运行时门槛未通过后要求测试T5。独立新实验，不训练，不覆盖旧结果。

官方权重原样加载、5次Euler（1,0.8,0.6,0.4,0.2，dt=-0.2），RTC强度1.25、EXP、horizon10。原8任务init27–31，朴素异步与RTC各自然/+100ms各40，共160正式；4套件task0/init13两方式两延迟，共16预检。

使用批准的首块等待修正：控制时钟启动前不以等待时长拒绝首块，保留实际耗时和不截断的延迟估计；运行中实际年龄>=10tick、全块过期、非有限、迟到率门槛均不变。20Hz、chunk50、执行10个有效动作后请求、batch1、单GPU模型、环境/模型分进程，线程1/4。旧T2采用旧首块拒绝逻辑，因此即使任务状态一致，也记录此代码配置差异；状态已用于T2评测，属于追加探索。

入口 `scripts/evaluate_t5_rtc_initial_wait.py`，模型适配器 `scripts/t5_rtc_policy.py`，共用隔离的 `rtc_initial_wait_controller.py`。远端输出 `/root/autodl-tmp/outputs/t5_rtc125_initialwait_v1`，日志 `/root/autodl-tmp/logs/t5_rtc125_initialwait_v1.log`，screen `t5-rtc125`。

本地编译、20项RTC测试、与T2完全一致的160正式及16预检调度检查通过。模型数值及实时门槛需远端实际验证。全部预检通过才运行正式；异常暂停，无自动重试或开关机。

```sh
cd /root/autodl-tmp/smolvla-flow
PYTHONPATH=src:. HF_HOME=/root/autodl-tmp/huggingface-cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py \
/root/autodl-tmp/venvs/smolvla-flow/bin/python -u -m scripts.evaluate_t5_rtc_initial_wait \
 --output /root/autodl-tmp/outputs/t5_rtc125_initialwait_v1 \
 --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
 --assets-dir /root/autodl-tmp/libero-assets \
 --source /root/autodl-tmp/outputs/s1_rtc_split_formal_v1 \
 --t2-source /root/autodl-tmp/outputs/t2_rtc125_pair_v1
```
