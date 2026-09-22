# T1 batch=1 交错供数执行记录

2026-09-17。用户确认在批量数值门槛失败后，采用多个环境进程共享一个模型，GPU 每次只处理一个样本。

## 当前状态

四套件 task0/init10 共 4 个原入口参考回合及 4 个新入口检查回合已完成。每个套件：噪声哈希序列一致、动作最大差 0、结束步数及成功判定一致。固定 rtol=1e-5、atol=1e-6 未放宽。

通过后程序自动进入 T1 400 回合正式评测。记录本页时已确认至少 55 个完成回合，未据部分结果推断最终成功率。正式运行的 8 个环境进程共享一份 GPU 模型。此配置尚未证明是所有工作进程数量中的最高吞吐配置，不宣称 GPU 已满载。

瞬时资源样本：GPU 64%，6210/24564 MiB，容器 memory.current 28475068416 bytes。不是平均利用率。后台每 5 秒由普通程序采集一次资源数据，不唤醒模型、不消耗对话 token。常规人工/助手进度检查不作分钟级轮询。

## 隔离及保护

- 官方 checkpoint 不变，flow_steps=1，动作块50，每环境执行1个动作后重新预测；不启用 RTC，不改变加载精度。
- 每回合独立 CUDA Generator，seed123，逐样本生成 1×50×32 噪声；每请求清空策略队列。
- 任务独占分配、同任务初始状态0–9按顺序执行，请求携带回合key和步数。每进程最多一个请求在途。
- 回合结果经检查后原子保存，收到保存确认后环境才进入下一回合。正常失败不重跑。
- 异常/非有限数值/资源限额超限停止，保留 error.json。恢复只处理未完成回合。
- 旧 T10/T5/S5/T2 正式回合不重跑、不覆盖。
- 正式开始前冻结8进程配置。运行中不为追求利用率修改进程数。

## 首轮实现问题

v1 的原入口参考检查完成，子进程因未继承 LIBERO `_assets_path_cache` 导致资产路径错误而停止。补全子进程资产路径和显式 CPU 随机种子后，在独立 v2 目录重试。复用 v1 原入口的4个参考回合，复用前核对协议、任务、权重文件、核心源码、软件版本、仿真资产哈希；没有重复参考回合来改变结果。

## 验证

- 三个入口脚本 Python 编译检查通过。
- `tests/test_t1_interleaved_gate.py` 8 项检查通过：正常、完成顺序变化、动作差异、噪声差异、成败差异、步数变化、缺套件、重复套件。
- 本机默认 Python 缺 numpy，测试改在远端既有环境执行，未为测试修改依赖。
- pilot_gate.json 中4组 max_abs均为0。该证据仅覆盖检查回合，不构成全部状态下逐值一致的保证。

## 路径及命令

- screen：`t1-interleaved`
- 日志：`/root/autodl-tmp/logs/t1_interleaved_v2.log`
- 输出：`/root/autodl-tmp/outputs/libero40_t1_interleaved_v2`
- 正式记录：输出内 `formal_run/formal/T1/`
- 结束时：`formal_run/result.json` 和 `formal_run/resources.json`
- 检查结果：输出内 `pilot_gate.json`
- 入口 SHA256：`f4e749c9164321706b976e99650ed2cdb3775da6757739e628ee7c26d9a32252`

```sh
cd /root/autodl-tmp/smolvla-flow
export PYTHONPATH=src:. OMP_NUM_THREADS=4 MUJOCO_GL=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/huggingface-cache
export DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py
/root/autodl-tmp/venvs/smolvla-flow/bin/python scripts/run_t1_interleaved.py \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --assets-dir /root/autodl-tmp/libero-assets \
  --output /root/autodl-tmp/outputs/libero40_t1_interleaved_v2 \
  --reference /root/autodl-tmp/outputs/libero40_t1_interleaved_v1 \
  --workers 8 --mode check-and-run
```

无错误的中断恢复使用同路径同代码 `--mode resume`；有 error.json 时须先定位，不自动清除错误文件。程序不自动开关机。

完成后下载新结果，用 `scripts/report_t1_interleaved.py --t1 artifacts/libero40_t1_interleaved_v2/formal_run --output artifacts/libero40_t1_comparison_v1` 重算。T1 并行合计吞吐、请求等待和 GPU 服务延迟单列，不进入旧单环境效率列。总运行时间从环境工作进程启动后计时，包含后续初始化、重置和视频写入；不包含加载模型和检查回合。
