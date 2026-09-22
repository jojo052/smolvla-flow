# RTC 引导强度单因素消融

用户于2026-09-19批准。固定 S1 010000.pt，不训练。只比较引导上限0、0.5、1；0使用朴素异步路径。原8任务，各init25–26，两种延迟，共96回合，每条件16回合。以seed123打乱任务/状态顺序，并轮换六种条件的顺序；每回合重置环境、策略随机流及队列。

## 不变配置

保留20Hz、S1一次积分、动作块50、执行10个有效动作后发起请求、实际过期裁剪、EXP、horizon10、夹爪处理、CPU/GPU进程拆分和batch1。保留迟到超过5ms的tick不超过5%的门槛。环境预算保持220/280/300/520，稳定10tick另计。每任务/条件保存首成功与首失败视频。

不沿用新状态上的历史对照，三个强度全部同批运行。历史结果只读，不覆盖。任一执行异常立即停止，没有自动重试、追加训练、追加搜索或开关机。

## 入口与来源

- 入口：`scripts/ablate_rtc_strength.py`
- screen：`rtc-strength-ablation`
- 输出：`/root/autodl-tmp/outputs/rtc_strength_ablation_v1`
- 日志：`/root/autodl-tmp/logs/rtc_strength_ablation_v1.log`
- 自动报告：输出目录下`report.json`，须满足96条唯一回合、状态/权重/强度身份一致。
- 启动前核对官方权重与处理器文件、S1权重、原执行器依赖文件哈希，核验8任务的27个初始状态均存在且前25个与原清单一致。
- 扫描服务器outputs中保存的回合/错误JSON；若发现init25或26已有执行记录则暂停。该扫描只说明保存的运行记录，不能证明不存在未保存的历史或预训练重叠。

## 运行命令

```sh
cd /root/autodl-tmp/smolvla-flow
export PYTHONPATH=src:.
export HF_HOME=/root/autodl-tmp/huggingface-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py
/root/autodl-tmp/venvs/smolvla-flow/bin/python -u -m scripts.ablate_rtc_strength \
  --stage run --output /root/autodl-tmp/outputs/rtc_strength_ablation_v1 \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --student /root/autodl-tmp/outputs/distill40_s1_formal_v1/training/010000.pt \
  --assets-dir /root/autodl-tmp/libero-assets \
  --source /root/autodl-tmp/outputs/s1_rtc_split_formal_v1
```

本地编译通过，96条唯一清单断言通过，现有RTC测试15项通过。实时循环不添加额外模型前向，沿用逐tick动作、边界跳变和请求指标；额外反事实动作比较如有必要应另做离线诊断，避免改变本轮延迟。

主要比较为0.5对1，以及二者各自对0的配对成功差异、按8任务重采样区间、任务完成效率与三类边界跳变。只有每任务2个初始状态，结论为探索性，不宣称新任务泛化。
