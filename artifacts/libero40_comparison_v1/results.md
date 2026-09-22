# 四套件 40 任务评估

未完成套件的成功率显示为待完成，不用部分回合估计正式结果。

| 模型 | Spatial | Object | Goal | Long | 宏平均 | rollout tick/s | 完整预测 P50/P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| smolvla_official (400/400) | 72.00% | 90.00% | 79.00% | 34.00% | 68.75% | 3.03 | 297.01/410.81 |
| act_public (400/400) | 25.00% | 11.00% | 3.00% | 14.00% | 13.25% | 51.31 | 9.50/10.63 |

## 逐任务

| 模型 | 套件 | task | 成功/回合 | Wilson 95% | 超时 |
|---|---|---:|---:|---|---:|
| smolvla_official | libero_spatial | 0 | 4/10 | 16.82% 至 68.73% | 6 |
| smolvla_official | libero_spatial | 1 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_spatial | 2 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_spatial | 3 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_spatial | 4 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_spatial | 5 | 1/10 | 1.79% 至 40.42% | 9 |
| smolvla_official | libero_spatial | 6 | 8/10 | 49.02% 至 94.33% | 2 |
| smolvla_official | libero_spatial | 7 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_spatial | 8 | 7/10 | 39.68% 至 89.22% | 3 |
| smolvla_official | libero_spatial | 9 | 4/10 | 16.82% 至 68.73% | 6 |
| smolvla_official | libero_object | 0 | 7/10 | 39.68% 至 89.22% | 3 |
| smolvla_official | libero_object | 1 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_object | 2 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_object | 3 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_object | 4 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_object | 5 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_object | 6 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_object | 7 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_object | 8 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_object | 9 | 8/10 | 49.02% 至 94.33% | 2 |
| smolvla_official | libero_goal | 0 | 4/10 | 16.82% 至 68.73% | 6 |
| smolvla_official | libero_goal | 1 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_goal | 2 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_goal | 3 | 2/10 | 5.67% 至 50.98% | 8 |
| smolvla_official | libero_goal | 4 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_goal | 5 | 8/10 | 49.02% 至 94.33% | 2 |
| smolvla_official | libero_goal | 6 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_goal | 7 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_goal | 8 | 10/10 | 72.25% 至 100.00% | 0 |
| smolvla_official | libero_goal | 9 | 8/10 | 49.02% 至 94.33% | 2 |
| smolvla_official | libero_10 | 0 | 1/10 | 1.79% 至 40.42% | 9 |
| smolvla_official | libero_10 | 1 | 2/10 | 5.67% 至 50.98% | 8 |
| smolvla_official | libero_10 | 2 | 1/10 | 1.79% 至 40.42% | 9 |
| smolvla_official | libero_10 | 3 | 9/10 | 59.58% 至 98.21% | 1 |
| smolvla_official | libero_10 | 4 | 0/10 | 0.00% 至 27.75% | 10 |
| smolvla_official | libero_10 | 5 | 8/10 | 49.02% 至 94.33% | 2 |
| smolvla_official | libero_10 | 6 | 8/10 | 49.02% 至 94.33% | 2 |
| smolvla_official | libero_10 | 7 | 0/10 | 0.00% 至 27.75% | 10 |
| smolvla_official | libero_10 | 8 | 0/10 | 0.00% 至 27.75% | 10 |
| smolvla_official | libero_10 | 9 | 5/10 | 23.66% 至 76.34% | 5 |
| act_public | libero_spatial | 0 | 2/10 | 5.67% 至 50.98% | 8 |
| act_public | libero_spatial | 1 | 5/10 | 23.66% 至 76.34% | 5 |
| act_public | libero_spatial | 2 | 2/10 | 5.67% 至 50.98% | 8 |
| act_public | libero_spatial | 3 | 5/10 | 23.66% 至 76.34% | 5 |
| act_public | libero_spatial | 4 | 2/10 | 5.67% 至 50.98% | 8 |
| act_public | libero_spatial | 5 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_spatial | 6 | 7/10 | 39.68% 至 89.22% | 3 |
| act_public | libero_spatial | 7 | 1/10 | 1.79% 至 40.42% | 9 |
| act_public | libero_spatial | 8 | 1/10 | 1.79% 至 40.42% | 9 |
| act_public | libero_spatial | 9 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_object | 0 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_object | 1 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_object | 2 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_object | 3 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_object | 4 | 2/10 | 5.67% 至 50.98% | 8 |
| act_public | libero_object | 5 | 3/10 | 10.78% 至 60.32% | 7 |
| act_public | libero_object | 6 | 2/10 | 5.67% 至 50.98% | 8 |
| act_public | libero_object | 7 | 1/10 | 1.79% 至 40.42% | 9 |
| act_public | libero_object | 8 | 2/10 | 5.67% 至 50.98% | 8 |
| act_public | libero_object | 9 | 1/10 | 1.79% 至 40.42% | 9 |
| act_public | libero_goal | 0 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_goal | 1 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_goal | 2 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_goal | 3 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_goal | 4 | 2/10 | 5.67% 至 50.98% | 8 |
| act_public | libero_goal | 5 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_goal | 6 | 1/10 | 1.79% 至 40.42% | 9 |
| act_public | libero_goal | 7 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_goal | 8 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_goal | 9 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_10 | 0 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_10 | 1 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_10 | 2 | 4/10 | 16.82% 至 68.73% | 6 |
| act_public | libero_10 | 3 | 2/10 | 5.67% 至 50.98% | 8 |
| act_public | libero_10 | 4 | 1/10 | 1.79% 至 40.42% | 9 |
| act_public | libero_10 | 5 | 3/10 | 10.78% 至 60.32% | 7 |
| act_public | libero_10 | 6 | 4/10 | 16.82% 至 68.73% | 6 |
| act_public | libero_10 | 7 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_10 | 8 | 0/10 | 0.00% 至 27.75% | 10 |
| act_public | libero_10 | 9 | 0/10 | 0.00% 至 27.75% | 10 |

## 完成时间

- libero40_public_v1: 2026-09-15T00:19:58.354597+08:00
- libero40_act_v1: 2026-09-15T01:14:40.902524+08:00

## 配对比较

```json
{
  "models": [
    "smolvla_official",
    "act_public"
  ],
  "counts": {
    "both_success": 43,
    "first_only": 232,
    "second_only": 10,
    "both_fail": 115
  },
  "first_minus_second": 0.555,
  "task_bootstrap_95": [
    0.445,
    0.655
  ],
  "bootstrap_replicates": 10000
}
```

## 效率

| 模型 | 混合 select_action 均值 ms | 分配显存峰值 GiB | 成功回合/小时（含失败与重置） |
|---|---:|---:|---:|
| smolvla_official | 311.527 | 1.472 | 33.14 |
| act_public | 0.610 | 0.215 | 60.83 |

## 解释边界

ACT 为 jamongsteak/act_libero 无语言条件公开权重，每次预测执行100动作；SmolVLA 为官方10 Flow steps、每步重新预测。训练数据版本与规模未控制，结果仅适用于这些checkpoint和原生执行配置，不能证明ACT架构普遍较弱。单任务专门训练的历史ACT结果不并入本轮。低ACT成功率的具体成因需要独立诊断，严格加载与有限动作检查不等同于全部语义正确性证明。

双方各400正式回合、初始状态0至9、任务与资产及共享环境代码一致。视频已回传，报告未对视频内容进行完整人工审阅。权重哈希在远端无卡实例再次核对通过。
