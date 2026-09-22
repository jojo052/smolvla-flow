# RTC 引导上限 1 的独立跟进评测

2026-09-19 用户批准只重跑异步＋RTC。复用 S1 `010000.pt`，只将 RTC 引导上限由 10 改为 1，不训练。

- 四套件 task 0、5，初始状态 20–24；自然延迟、额外 100ms 各 40 回合，共 80。
- 保留 20Hz、执行10个有效动作后请求、50动作块、一次积分、batch1、split-process、实际过期裁剪和原时间门控。
- 使用原调度清单过滤 RTC 条目，保留相对顺序。运行前核对权重、处理器、原代码哈希和低权重16回合检查结果。
- 新目录 `/root/autodl-tmp/outputs/s1_rtc_weight1_formal_v1`。旧目录 `s1_rtc_split_formal_v1` 只读；所有 240 条历史回合文件哈希写入新 manifest，报告时重新校验。
- 保留权重10的原始结果；同步、朴素异步使用历史对照。明确不同运行时间、已知测试条件以及看到旧结果后的跟进实验限制。
- 异常暂停，不自动重试，不自动开关机。完成时自动生成 `report.json`，包含两种延迟的结果及相对历史异步对照的配对比较。

入口：`scripts/evaluate_rtc_weight1.py`；后台 screen：`rtc-weight1-formal`；日志：`/root/autodl-tmp/logs/s1_rtc_weight1_formal_v1.log`。

运行命令（原有 HF 离线、EGL、线程及冻结入口环境变量保持不变）：

```sh
/root/autodl-tmp/venvs/smolvla-flow/bin/python -u -m scripts.evaluate_rtc_weight1 \
  --stage run \
  --output /root/autodl-tmp/outputs/s1_rtc_weight1_formal_v1 \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --student /root/autodl-tmp/outputs/distill40_s1_formal_v1/training/010000.pt \
  --assets-dir /root/autodl-tmp/libero-assets \
  --source /root/autodl-tmp/outputs/s1_rtc_split_formal_v1 \
  --check-output /root/autodl-tmp/outputs/rtc_low_guidance_check_v1
```

本地验证：新入口编译通过，既有 RTC 单元测试 15 项通过；清单断言为80个唯一RTC条目、每延迟40个、初始状态集合20–24。结果以新目录实际完成回合及报告为准，不将启动等同于完成。
