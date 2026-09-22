# 四套件 40 任务评估

未完成套件的成功率显示为待完成，不用部分回合估计正式结果。

| 模型 | Spatial | Object | Goal | Long | 宏平均 | rollout tick/s | 完整预测 P50/P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| smolvla_official (400/400) | 72.00% | 90.00% | 79.00% | 34.00% | 68.75% | 3.03 | 297.01/410.81 |

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

ACT：公开权重尚未通过核验，未运行；不形成两模型优劣结论。
