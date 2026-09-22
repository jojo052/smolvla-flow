# 四套件公开模型 benchmark v1

本实验独立于旧 task34 结果。配置：`configs/libero40_public_v1.json`。
完整协议、权重与处理器哈希、任务 BDDL/初始状态哈希、代码指纹和依赖版本写入输出目录 `manifest.json`。

## ACT 准入核查（2026-09-14）

| 候选 | 已知证据 | 未解决问题 | 本轮状态 |
|---|---|---|---|
| 本项目 ACT 050000 | task34 训练和 30 回合记录 | 未覆盖其他任务 | 排除 |
| [DILLO](https://github.com/MaxPappa/DILLO) | README 声明提供四套件权重 | 评估代码使用 GMM head，偏离选定的标准 ACT 范围 | 排除 |
| [jamongsteak/act_libero](https://huggingface.co/jamongsteak/act_libero) | 固定版本8744e679已通过用户授权的镜像下载；8个文件校验、CPU严格加载与处理器检查通过，模型卡40条任务全部匹配 | 无语言条件，原生执行100动作；数据集revision未固定；GPU闭环检查未执行 | CPU核验通过，待独立闭环评估；详见 artifacts/libero40_act_audit_v1/report.md |
| [OpenRAL/rskill-act-franka_panda-libero-fp32](https://huggingface.co/OpenRAL/rskill-act-franka_panda-libero-fp32) | 公共模型页面 | 本次未核验四套件覆盖和完整加载契约 | 未准入 |

按启动时用户选定的回退方案，本轮执行官方 SmolVLA 400 回合。随后用户授权通过镜像下载ACT，并完成CPU核验。现有冻结的SmolVLA运行继续保持原样。ACT不训练、不用task34权重冒充多任务权重。后续闭环比较应建立独立ACT运行manifest及明确关联的比较报告，不能修改已冻结输出目录来补写ACT。

## 运行协议

- Spatial/Object/Goal/Long 每套件 10 任务，动作预算分别 220/280/300/520。
- 官方初始状态索引 0–9；每套件 task0 的 index10 专用于检查，不计入正式结果。
- 10 个无动作稳定步，策略与环境 seed=123；每回合重置队列。
- SmolVLA 固定官方权重、10 Flow steps、每个动作重新生成完整 50 动作预测。
- 使用 checkpoint pre/postprocessor；环境侧 LiberoProcessorStep 翻转 H/W，拼接 8D 状态。
- 动作直接使用 checkpoint 反归一化结果；不加入旧项目的 normalized-action 夹爪滞回处理。该项为原生评估适配，与历史 task34 的额外处理不同。
- 保留 LeRobot reset 初始状态与 10 步稳定行为；同步 step 省去成功后的隐式 reset。
- 每个模型单环境串行，完整预测和 select_action 均用 CUDA synchronize 包围计时。
- rollout 延迟不含 reset；成功回合/小时包含全部成功、失败回合及 reset，排除视频编码与加载模型。视频采集拷贝计入 rollout，记录条件随首个成功/失败视频是否已存在而变化。
- 成功率不用于检查阶段选模；动作有限性、尺寸、状态索引、严格权重加载是准入条件。

## 命令

在独立部署目录中执行，需已安装项目与 LeRobot/LIBERO 依赖：

```bash
PYTHONPATH=src:. MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_libero40.py \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --assets-dir /root/autodl-tmp/libero-assets \
  --output /root/autodl-tmp/outputs/libero40_public_v1 \
  --phase all
```

`--phase pilot` 仅运行检查；`--phase formal` 必须已有四个有效检查回合；`--phase report` 从已有回合离线重建结果，不加载模型。
再次运行相同命令仅跳过已完整保存并校验通过的回合。文件锁阻止并发写入；协议或代码指纹发生变化会拒绝续跑。执行错误保存到 `errors/` 并停止，不当作模型失败。发生错误后必须人工检查，再显式传入 `--acknowledge-errors`；协议和代码变化仍然拒绝续跑。

输出包含 `status.json`、`estimate.json`、`results.json`、`results.md`、逐回合 JSON、首个成功/失败视频。未完成套件不输出正式成功率。每任务 Wilson 区间，双模型配对工具仅在 800 回合完整且初始状态哈希一致时输出 task-bootstrap 95% 区间。

## 验证

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -p test_benchmark40.py -v
```

单元测试覆盖 400 个唯一身份、检查集隔离、冻结配置、原子保存、错误与超时区别、部分结果、重复计数、失败耗时计入、Wilson 区间及配对 bootstrap。
真实 GPU 检查另外验证全 chunk `[1,50,7]`、每步一次预测、8D 状态及双 256×256 图像。检查通过不代表任务成功率达到某个分数。
