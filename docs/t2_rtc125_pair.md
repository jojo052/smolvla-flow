# 官方 T2：朴素异步与 RTC 1.25 配对实验

2026-09-20 用户批准立即验证两组，并明确选择RTC引导强度1.25。固定官方权重，两次Euler更新，t=1、0.5，dt=-0.5，不加载S1、不训练。

## 协议

- 原8任务，每套件task0、5；官方初始状态27–31，与此前调参的20–26分离。
- 两方式（朴素异步、RTC），两延迟（自然、额外100ms），每条件40回合，总160。
- 控制20Hz，稳定10tick；chunk50，每接纳新块执行10个有效动作后发起下一请求，观测年龄裁剪过期前缀，RTC EXP、horizon10。
- GPU batch1，单模型，环境与模型分进程；控制进程PyTorch1线程、模型4线程。
- 原动作、归一化、队列、视频及迟到率门槛不变，任一异常暂停，不自动重启或关机。
- 4个套件task0/init13，两方式两延迟，共16个预检回合，不计入正式成绩；另做固定输入两步及RTC有效性检查。
- 每回合种子123。seed123打乱任务状态对、轮换四条件顺序。
- 保存原始请求/动作、前向次数、视频、模型文件指纹。共享控制器的S1身份字段在持久化之前移除，明确记录官方T2身份。

预启动核对原代码、官方权重与处理器哈希；核对原初始状态哈希并保存新状态哈希，扫描已保存的回合及错误记录，若已有27–31执行记录则暂停。该扫描不保证无未保存历史或预训练数据重叠。

## 入口和运行位置

- `scripts/t2_rtc_policy.py`：独立T2适配器；没有学生权重覆盖。
- `scripts/evaluate_t2_rtc.py`：run/resume/report、验证身份、独立配对报告。
- 远端输出：`/root/autodl-tmp/outputs/t2_rtc125_pair_v1`
- screen：`t2-rtc125`
- 日志：`/root/autodl-tmp/logs/t2_rtc125_pair_v1.log`

```sh
cd /root/autodl-tmp/smolvla-flow
PYTHONPATH=src:. HF_HOME=/root/autodl-tmp/huggingface-cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py \
/root/autodl-tmp/venvs/smolvla-flow/bin/python -u -m scripts.evaluate_t2_rtc \
 --output /root/autodl-tmp/outputs/t2_rtc125_pair_v1 \
 --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
 --assets-dir /root/autodl-tmp/libero-assets \
 --source /root/autodl-tmp/outputs/s1_rtc_split_formal_v1
```

本地编译通过，现有RTC测试15项通过；160唯一回合、每条件40、16检查回合、随机顺序可复现断言通过。GPU数值、模型身份和实际调度验收由远端preflight与逐回合检查决定，不能用本地测试替代。

## 解释边界

主要比较是同一T2的RTC减去朴素异步，报告配对成败、8任务bootstrap区间和效率。新初始状态与S1实验不同，不能直接把跨实验差值解释成蒸馏造成的RTC退化。正结果只能支持进一步研究RTC感知蒸馏；负结果只涉及此模型、任务和固定RTC配置，不能推出所有RTC无效。不根据正式成绩追加搜索。
