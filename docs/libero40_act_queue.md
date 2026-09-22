# ACT 串行评估交接

2026-09-14 用户授权 ACT 四套件评估排在当前官方 SmolVLA 评估之后，无新训练。

远端目录为 `/root/autodl-tmp/benchmarks/libero40_public_v1`。
队列入口 `scripts/queue_libero40_act.py`，screen `libero40-act-queue`。
队列及 ACT 日志 `/root/autodl-tmp/logs/libero40_act_queue.log`。
独立结果 `/root/autodl-tmp/outputs/libero40_act_v1`。

前序必须正常完成 400 个已验证正式回合并释放运行锁，官方权重哈希一致，GPU 无计算进程。队列再核对 ACT 的八个审计文件哈希，启动独立 ACT 入口。前序错误、哈希不一致、GPU 被占用时不会启动 ACT，也不会重启前序或关机。

ACT 使用 jamongsteak/act_libero，版本 8744e679f4038386070ff70723edab9bb6228bd3。该 checkpoint 无语言条件，原生预测100个动作并执行100个动作，无 temporal ensemble。结果只描述这个公开 checkpoint。

四套件各 task0 初始状态10作为 pilot，检查严格加载、有限动作、输入输出和指标契约，成功率不作为准入门槛。通过后执行40任务各10回合，初始状态0至9。独立manifest、日志、视频与逐回合结果保留，不修改冻结的 SmolVLA 代码及输出。

CPU测试：ACT和既有SmolVLA适配、统计模块共20 tests及7 subtests通过。包括101个执行tick恰好2次完整ACT预测、回合清队列、非有限动作拒绝、初始状态索引及前序完成门槛。GPU闭环pilot排在前序之后执行，尚不代表已经通过。

回访 libero40 已更新为跟进双方完成及独立配对汇总；正常逐回合推进不通知，套件完成、交接或异常时通知。队列本身在远端运行，不依赖本地SSH持续在线。服务器关机或重启会终止队列，未配置开机自启。
