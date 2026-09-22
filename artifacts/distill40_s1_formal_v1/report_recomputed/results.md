# T2→S1 六模型对照

| 模型 | Spatial | Object | Goal | Long | 40任务宏平均 | 历史单环境预测 P50/P95 ms | 历史吞吐 tick/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| T10 | 72.00% | 90.00% | 79.00% | 34.00% | 68.75% | 297.01/410.81 | 3.03 |
| T5 | 76.00% | 86.00% | 75.00% | 44.00% | 70.25% | 162.71/222.06 | 5.33 |
| S5 | 76.00% | 93.00% | 75.00% | 37.00% | 70.25% | 162.00/221.12 | 5.37 |
| T2 | 79.00% | 94.00% | 80.00% | 45.00% | 74.50% | 92.46/123.78 | 8.55 |
| T1 | 73.00% | 65.00% | 72.00% | 40.00% | 62.50% | 并行指标单列 | 并行指标单列 |
| S1 | 80.00% | 88.00% | 76.00% | 41.00% | 71.25% | 并行指标单列 | 并行指标单列 |

## 并行效率与配对比较

```json
{
  "T1_parallel_efficiency": {
    "end_to_end_seconds": 6997.72778050974,
    "aggregate_tick_per_s": 13.332613517755878,
    "success_per_hour": 128.61317676670765,
    "server_prediction_latency": {
      "mean_ms": 72.13518398112213,
      "p50_ms": 69.36631351709366,
      "p95_ms": 91.60519763827324,
      "count": 93298
    },
    "request_wait_latency": {
      "mean_ms": 430.3792805062881,
      "p50_ms": 386.85017079114914,
      "p95_ms": 902.8834030032158,
      "count": 93298
    },
    "mean_gpu_util": 46.579453067257944,
    "peak_total_gpu_mib": 6725.0,
    "includes_rendering_gpu_use": true
  },
  "S1_parallel_efficiency": {
    "end_to_end_seconds": 6216.494450703263,
    "aggregate_tick_per_s": 13.863601211813233,
    "success_per_hour": 165.04478659736117,
    "server_prediction_latency": {
      "mean_ms": 69.31590826631093,
      "p50_ms": 68.22453811764717,
      "p95_ms": 73.3342319726944,
      "count": 86183
    },
    "request_wait_latency": {
      "mean_ms": 418.5248442581536,
      "p50_ms": 377.6530921459198,
      "p95_ms": 874.6944963932037,
      "count": 86183
    },
    "mean_gpu_util": 48.571072319201996,
    "peak_total_gpu_mib": 6726.0,
    "includes_rendering_gpu_use": true
  },
  "S1_minus_T1": {
    "models": [
      "S1",
      "T1"
    ],
    "counts": {
      "both_success": 225,
      "first_only": 60,
      "second_only": 25,
      "both_fail": 90
    },
    "first_minus_second": 0.0875,
    "task_bootstrap_95": [
      0.017499999999999998,
      0.165
    ],
    "bootstrap_replicates": 10000
  },
  "S1_minus_T2": {
    "models": [
      "S1",
      "T2"
    ],
    "counts": {
      "both_success": 267,
      "first_only": 18,
      "second_only": 31,
      "both_fail": 84
    },
    "first_minus_second": -0.0325,
    "task_bootstrap_95": [
      -0.0675,
      6.938893903907229e-19
    ],
    "bootstrap_replicates": 10000
  }
}
```

S1 教师目标采用批量计算，与历史单样本 T2 存在数值差异。使用已知测试初始状态，不作为未触及测试集的泛化证明。
教师和历史模型没有追加正式评测。逐任务 Wilson 区间、失败类型和配对重采样区间见 results.json。
