# SmolVLA Flow Matching: low-step distillation and real-time execution

This repository studies two engineering questions for a vision-language-action policy:

1. How much Flow Matching inference can be reduced before closed-loop performance degrades?
2. Does action-chunk asynchronous execution help more than RTC guidance for real-time control?

The main benchmark is a fixed 40-task LIBERO evaluation. The repository keeps the raw episodes, manifests, videos, checkpoints, and report inputs needed to recompute the tables.

中文版：[README.zh-CN.md](README.zh-CN.md)

## Main result

The strongest single-environment low-step result is the official SmolVLA checkpoint evaluated with two Flow steps (T2): 74.50% macro success over 400 episodes. The public ACT checkpoint is included as an external baseline: it reaches 13.25% on the same 40-task state set while running at 51.31 tick/s, but it uses its native 100-action chunk and has no language conditioning. The distilled one-step student (S1) recovers 8.75 percentage points over the direct one-step model (T1), but remains 3.25 points below T2. The five-step student (S5) matches direct five-step inference (T5) in macro success, so this run does not show a five-step distillation gain over direct step reduction.

For real-time execution, naive asynchronous chunk execution is the reliable improvement. RTC guidance reduces chunk-boundary motion jumps, but the tested S1 RTC configurations do not provide a stable success-rate improvement.

## LIBERO dataset and evaluation protocol

This project uses the four ten-task LIBERO suites: Spatial, Object, Goal, and LIBERO-10. Together they provide 40 task identities. Each task is identified by its language instruction, BDDL definition, task-specific initial-state file, and maximum control budget.

The full dataset and protocol description is in [docs/libero_dataset.md](docs/libero_dataset.md). The frozen task and state hashes are in [the 40-task manifest](artifacts/libero40_public_v1/manifest.json).

The formal benchmark uses the official initial-state indices `0–9` for every task:

```text
40 tasks × 10 initial states = 400 episodes per model
```

Fixed conditions are policy and environment seed `123`, ten settle steps, two 256×256 RGB observations, an 8D robot state, 7D actions, chunk length 50, one executed action per prediction, and no RTC. Maximum control budgets are 220, 280, 300, and 520 ticks for Spatial, Object, Goal, and LIBERO-10 respectively.

Distillation observations come from `HuggingFaceVLA/libero`, revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`. The recorded data manifest hash is `9471cb4559782463642d593a6d82ec9a257ed3130e9f11dc40ec3966cf5bcf5e`. The distillation objective uses teacher-generated action targets; S1 does not add an expert-action behavior-cloning loss. Training data and simulator evaluation data are reported separately.

Official references: [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO documentation](https://lifelong-robot-learning.github.io/LIBERO/html/index.html), and [LIBERO on Hugging Face](https://huggingface.co/datasets/HuggingFaceVLA/libero).

## Distillation and low-step results

All rows below contain 400 completed episodes, 100 per suite. T10, T5, S5, and T2 use the historical single-environment runner. T1 and S1 use the separately audited eight-environment interleaved runner; their parallel throughput is reported separately from the single-environment rows.

| Model | Definition | Spatial | Object | Goal | Long | Macro success | Prediction P50/P95 | Rollout throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T10 | Official checkpoint, 10 steps | 72/100 | 90/100 | 79/100 | 34/100 | **68.75%** | 297.0 / 410.8 ms | 3.03 tick/s |
| T5 | Official checkpoint, direct 5 steps | 76/100 | 86/100 | 75/100 | 44/100 | **70.25%** | 162.7 / 222.1 ms | 5.33 tick/s |
| S5 | T10 → S5 distillation | 76/100 | 93/100 | 75/100 | 37/100 | **70.25%** | 162.0 / 221.1 ms | 5.37 tick/s |
| T2 | Official checkpoint, direct 2 steps | 79/100 | 94/100 | 80/100 | 45/100 | **74.50%** | 92.5 / 123.8 ms | 8.55 tick/s |
| T1 | Official checkpoint, direct 1 step | 73/100 | 65/100 | 72/100 | 40/100 | **62.50%** | parallel service: 69.4 / 91.6 ms | parallel: 13.33 tick/s |
| S1 | T2 → S1 distillation | 80/100 | 88/100 | 76/100 | 41/100 | **71.25%** | parallel service: 68.2 / 73.3 ms | parallel: 13.86 tick/s |
| ACT | Public `jamongsteak/act_libero`, native 100-action chunk | 25/100 | 11/100 | 3/100 | 14/100 | **13.25%** | 9.5 / 10.6 ms | 51.31 tick/s |

Interpretation:

- S5 and T5 have the same macro success. S5 is 7 points higher on Object and 7 points lower on Long.
- S1 improves over T1 by 8.75 points, showing recovery at one step, but is 3.25 points below T2.
- T2 is the best current low-step success result under the primary 40-task protocol.
- ACT is much faster in its native chunked runner, but its public checkpoint is far below the SmolVLA rows on this 40-task evaluation. This is a checkpoint comparison, not an equal-training or equal-language-conditioning comparison.

ACT uses the public `jamongsteak/act_libero` revision `8744e679f4038386070ff70723edab9bb6228bd3`, with model SHA-256 `ae1d034c4d30879773a1f67590194a7ce5246c8466eb46db299696bfab2bb247`. Its native configuration predicts and executes 100 actions per chunk, has no temporal ensemble, and is not language-conditioned. The ACT row is therefore a required baseline comparison, while its prediction latency and tick/s are not a like-for-like replacement for the SmolVLA single-action re-prediction measurements.

The full ACT versus official T10 paired table is in [artifacts/libero40_comparison_v1/results.md](artifacts/libero40_comparison_v1/results.md). On matched task and initial-state records, T10 succeeds alone 232 times, ACT succeeds alone 10 times, both succeed 43 times, and both fail 115 times. The paired T10 minus ACT difference is 55.5 percentage points, with a task-bootstrap 95% interval of 44.5 to 65.5 points.

The source data and paired comparisons are in [the recomputed result JSON](artifacts/distill40_s1_formal_v1/report_recomputed/results.json).

## RTC results

RTC results use S1 on eight tasks, initial-state indices 20–24, 40 episodes per condition. They are a separate real-time study, not additional samples for the 400-episode benchmark.

### Natural inference latency

| Execution mode | Success | Mean episode time | Success/hour | Queue empty | Prediction P50/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synchronous chunk | 13/40 (32.50%) | 18.75 s | 62.40 | 22.69% | 102.42 / 126.17 ms |
| Naive asynchronous | **33/40 (82.50%)** | 14.17 s | **209.60** | 0% | 101.79 / 118.33 ms |
| Async + RTC 1.25 | 24/40 (60.00%) | 15.20 s | 142.08 | 0% | 108.40 / 120.58 ms |

### Additional 100 ms latency

| Execution mode | Success | Mean episode time | Success/hour | Queue empty | Prediction P50/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synchronous chunk | 3/40 (7.50%) | 20.65 s | 13.08 | 32.69% | 109.68 / 130.01 ms |
| Naive asynchronous | 27/40 (67.50%) | 15.42 s | 157.57 | 0% | 114.43 / 132.34 ms |
| Async + RTC 1.25 | 27/40 (67.50%) | 14.45 s | 168.11 | 0% | 117.99 / 130.94 ms |

RTC reduces mean translational chunk-boundary jumps from 0.26093 to 0.10666 under natural latency, and from 0.28742 to 0.11170 with the extra 100 ms delay. It did not produce a stable success-rate gain in this S1 study. The full analysis, including guidance-1.0, guidance-10, and T2 RTC follow-ups, is in [docs/final_results.md](docs/final_results.md).

## Reproduction and provenance

- Primary protocol: [configs/libero40_public_v1.json](configs/libero40_public_v1.json)
- Dataset and task mapping: [docs/libero_dataset.md](docs/libero_dataset.md)
- Distillation execution record: [docs/s1_execution.md](docs/s1_execution.md)
- RTC 1.25 result: [docs/rtc_weight125_results.md](docs/rtc_weight125_results.md)
- T2 RTC pair: [docs/t2_rtc125_pair.md](docs/t2_rtc125_pair.md)
- Complete narrative report: [docs/final_results.md](docs/final_results.md)

The old task34, 30-seed experiment is retained as a historical appendix in [docs/legacy_task34.md](docs/legacy_task34.md). It is not mixed into the 40-task tables above.

```bash
python -m pip install -e ".[dev,plot]"
python -m pytest -q
```

The final archive contains raw result JSON, manifests, videos, checkpoints, and SHA-256 inventories. Credentials, virtual environments, model caches, and dataset caches are excluded.
