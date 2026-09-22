# 四套件 40 任务评估

未完成套件的成功率显示为待完成，不用部分回合估计正式结果。

| 模型 | Spatial | Object | Goal | Long | 宏平均 | rollout tick/s | 完整预测 P50/P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| act_public (400/400) | 25.00% | 11.00% | 3.00% | 14.00% | 13.25% | 51.31 | 9.50/10.63 |

## 逐任务

| 模型 | 套件 | task | 成功/回合 | Wilson 95% | 超时 |
|---|---|---:|---:|---|---:|
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

ACT：公开无语言条件模型，原生每次预测执行100个动作；本表不含SmolVLA配对比较。
