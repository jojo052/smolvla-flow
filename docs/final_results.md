# Final experimental report

This report is the reproducible summary of the 40-task LIBERO benchmark and the separate real-time execution study. The dataset and fixed protocol are described in [libero_dataset.md](libero_dataset.md). Raw inputs remain in `artifacts/`; the tables below are generated from those files rather than from manually selected episodes.

## Executive summary

The direct two-step official checkpoint, T2, is the strongest low-step policy in the primary 40-task benchmark at 74.50% macro success. The public ACT checkpoint is also included as an external baseline. It reaches 13.25% on the same 400-episode state set and runs at 51.31 tick/s in its native 100-action chunk configuration. Distillation is still useful at the one-step boundary: S1 improves direct one-step T1 by 8.75 percentage points, from 62.50% to 71.25%. It does not recover T2, which is 3.25 points higher.

The five-step student S5 matches direct five-step inference T5 at 70.25%. Its task profile changes, with a gain on Object and a loss on LIBERO-10, but there is no macro success gain. The result supports the narrower claim that distillation can recover part of the capability lost by reducing the number of Flow steps. It does not support a claim that the student is better than direct step reduction under this training recipe.

For real-time execution, naive asynchronous chunk execution is the most reliable improvement measured here. On the S1 eight-task RTC study, it reaches 33/40 under natural latency versus 13/40 for the synchronous waiting baseline. RTC guidance reduces chunk-boundary motion jumps, but its success-rate effect is not consistently positive.

## Primary 40-task benchmark

Each row has 400 completed episodes, with 100 episodes in each of Spatial, Object, Goal, and LIBERO-10. T10, T5, S5, and T2 were measured with the historical single-environment runner. T1 and S1 used the separately audited eight-environment interleaved runner; their service latency and aggregate throughput are therefore reported with a separate label.

| Model | Definition | Spatial | Object | Goal | Long | Macro success | Prediction P50/P95 | Rollout throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T10 | Official checkpoint, 10 Flow steps | 72/100 | 90/100 | 79/100 | 34/100 | **68.75%** | 297.0 / 410.8 ms | 3.03 tick/s |
| T5 | Official checkpoint, direct 5 steps | 76/100 | 86/100 | 75/100 | 44/100 | **70.25%** | 162.7 / 222.1 ms | 5.33 tick/s |
| S5 | T10 to S5 distillation | 76/100 | 93/100 | 75/100 | 37/100 | **70.25%** | 162.0 / 221.1 ms | 5.37 tick/s |
| T2 | Official checkpoint, direct 2 steps | 79/100 | 94/100 | 80/100 | 45/100 | **74.50%** | 92.5 / 123.8 ms | 8.55 tick/s |
| T1 | Official checkpoint, direct 1 step | 73/100 | 65/100 | 72/100 | 40/100 | **62.50%** | parallel service: 69.4 / 91.6 ms | parallel: 13.33 tick/s |
| S1 | T2 to S1 distillation | 80/100 | 88/100 | 76/100 | 41/100 | **71.25%** | parallel service: 68.2 / 73.3 ms | parallel: 13.86 tick/s |
| ACT | Public `jamongsteak/act_libero`, native 100-action chunk | 25/100 | 11/100 | 3/100 | 14/100 | **13.25%** | 9.5 / 10.6 ms | 51.31 tick/s |

The recomputed source is [`artifacts/distill40_s1_formal_v1/report_recomputed/results.json`](../artifacts/distill40_s1_formal_v1/report_recomputed/results.json). The 40-row task table, Wilson intervals, failure counts, and paired records are in the same report directory.

### Distillation comparisons

- **S5 versus T5:** macro difference 0 percentage points. S5 is 7 points higher on Object and 7 points lower on LIBERO-10; Spatial and Goal are unchanged.
- **S1 versus T1:** macro difference +8.75 points. The suite changes are +7 points Spatial, +23 Object, +4 Goal, and +1 Long.
- **S1 versus T2:** macro difference -3.25 points. On matched task and initial-state records, both succeed 267 times, both fail 84 times, S1 alone succeeds 18 times, and T2 alone succeeds 31 times. The task-bootstrap interval for the success-rate difference is approximately [-6.75, 0] percentage points.

### ACT baseline comparison

The ACT row is a real 40-task baseline, not the historical task34-only ACT result. It uses the public `jamongsteak/act_libero` checkpoint at revision `8744e679f4038386070ff70723edab9bb6228bd3`, with model SHA-256 `ae1d034c4d30879773a1f67590194a7ce5246c8466eb46db299696bfab2bb247`. The audit found strict loading with the expected two 256x256 images, 8D state, and 7D action. Its native config predicts 100 actions and executes 100 actions per chunk, without temporal ensemble and without language conditioning. This explains why its 9.5 ms full prediction and 51.31 tick/s are not directly interchangeable with the SmolVLA measurements, which re-predict after each executed action. The valid comparison is the completed 400-episode success result under the frozen task and initial-state set, with the execution-policy caveat stated explicitly.

On matched task and initial-state records, T10 succeeds alone 232 times, ACT succeeds alone 10 times, both succeed 43 times, and both fail 115 times. The paired T10 minus ACT difference is 55.5 percentage points, with a task-bootstrap 95% interval of 44.5 to 65.5 points. The complete ACT comparison, including all 40 task rows and Wilson intervals, is [`artifacts/libero40_comparison_v1/results.md`](../artifacts/libero40_comparison_v1/results.md).

These comparisons answer different questions. S5 versus T5 tests whether a student recovers or improves direct step reduction at the same nominal step count. S1 versus T1 tests one-step capability recovery. S1 versus T2 measures the remaining gap to the two-step official teacher. None of these results imply a universal ranking outside this checkpoint, task set, and protocol.

### Efficiency interpretation

T2 reduces the prediction P50 from 297.0 ms for T10 to 92.5 ms and increases the historical rollout rate from 3.03 to 8.55 tick/s. S1 was evaluated with the interleaved executor: aggregate rates were 13.86 tick/s and 165.04 successful episodes/hour, with mean GPU utilization 48.57%. T1 was 13.33 tick/s and 128.61 successful episodes/hour, with 46.58% mean GPU utilization. These aggregate parallel numbers must not be inserted into the single-environment latency column for T10, T5, S5, or T2.

## RTC and asynchronous execution

The RTC study uses S1 on eight selected tasks, initial-state indices 20-24, and 40 episodes per condition. It is a separate real-time study, not additional samples for the primary 400-episode benchmark. Historical controls were reused and were not run concurrently. The selected RTC guidance value is 1.25; guidance 1.0 and the original guidance 10.0 are sensitivity and failure diagnostics.

### S1, guidance 1.25

| Execution mode | Natural latency success | Natural mean episode | Natural success/hour | +100 ms success | +100 ms mean episode | +100 ms success/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synchronous chunk | 13/40 (32.50%) | 18.75 s | 62.40 | 3/40 (7.50%) | 20.65 s | 13.08 |
| Naive asynchronous | **33/40 (82.50%)** | 14.17 s | **209.60** | 27/40 (67.50%) | 15.42 s | 157.57 |
| Async + RTC 1.25 | 24/40 (60.00%) | 15.20 s | 142.08 | 27/40 (67.50%) | 14.45 s | 168.11 |

The full source is [`artifacts/s1_rtc_weight125_formal_v1/report.json`](../artifacts/s1_rtc_weight125_formal_v1/report.json) and the detailed table is in [`docs/rtc_weight125_results.md`](rtc_weight125_results.md). Under natural latency, the paired RTC minus async difference is -22.5 points, with task-bootstrap interval [-40, -5]. Under +100 ms, the difference is 0 points, with interval [-7.5, +7.5].

RTC does improve smoothness in this setup. Mean translational chunk-boundary jump decreases from 0.26093 to 0.10666 under natural latency and from 0.28742 to 0.11170 with +100 ms. The same run shows no stable success-rate improvement. Guidance 1.0 is less harmful than guidance 1.25, but remains below async: 29/40 versus 33/40 at natural latency, and 25/40 versus 27/40 with +100 ms. Guidance 10.0 is a clear negative control: 1/40 natural and 16/40 with +100 ms, with much larger jumps.

### T2 RTC follow-up

The T2 guidance-1.25 pair is a separate eight-task, initial-state 27-31 study. Natural latency is 32/40 for RTC versus 33/40 for async; with +100 ms it is 24/40 versus 27/40. The paired differences are -2.5 and -7.5 points respectively. RTC reduces mean translational jumps from 0.24548 to 0.10387 naturally and from 0.26578 to 0.10460 with +100 ms, but does not improve success in this sample.

T10 and T5 RTC runs did not produce valid formal comparisons. T10 stopped at the runtime gate during the initial-wait follow-up, and T5 stopped during preflight. They are archived as incomplete evidence, not counted as zero-success results. See [`docs/t10_rtc125_pair.md`](t10_rtc125_pair.md) and [`docs/t5_rtc125_pair.md`](t5_rtc125_pair.md).

## What the results support

1. Reducing Flow steps is a strong compute and latency lever. In this benchmark, direct T2 is both faster than T10 and the highest-success low-step row.
2. Distillation has a measurable role at the one-step boundary. S1 recovers most of the direct T1 loss, even though it does not reach T2.
3. The current S5 training recipe does not beat direct T5. Its value is capability recovery under a lower-step student formulation, not a demonstrated macro-success gain.
4. Asynchronous execution is more reliable than waiting for a new chunk. RTC reduces discontinuities, but the tested guidance and timing assumptions do not produce a stable success-rate gain.

The report does not claim that direct step reduction always wins, that RTC is generally harmful, or that this benchmark proves zero-shot generalization. Formal states 0-9 are a fixed comparison set; RTC follow-ups use known task families and different state indices.

## Provenance and reproducibility

- Primary protocol: [`configs/libero40_public_v1.json`](../configs/libero40_public_v1.json)
- Frozen task and state manifest: [`artifacts/libero40_public_v1/manifest.json`](../artifacts/libero40_public_v1/manifest.json)
- Dataset and integrity description: [`docs/libero_dataset.md`](libero_dataset.md)
- S1 training and selection: [`artifacts/distill40_s1_formal_v1/`](../artifacts/distill40_s1_formal_v1/)
- S1 checkpoint: [`artifacts/distill40_s1_formal_v1/training/010000.pt`](../artifacts/distill40_s1_formal_v1/training/010000.pt)
- Primary result source: [`artifacts/distill40_s1_formal_v1/report_recomputed/`](../artifacts/distill40_s1_formal_v1/report_recomputed/)
- RTC result sources: [`artifacts/s1_rtc_weight125_formal_v1/`](../artifacts/s1_rtc_weight125_formal_v1/), [`artifacts/s1_rtc_weight1_formal_v1/`](../artifacts/s1_rtc_weight1_formal_v1/), and [`artifacts/t2_rtc125_pair_v1/`](../artifacts/t2_rtc125_pair_v1/)
- ACT baseline source and paired table: [`artifacts/libero40_act_v1/`](../artifacts/libero40_act_v1/) and [`artifacts/libero40_comparison_v1/`](../artifacts/libero40_comparison_v1/)

The selected S1 checkpoint SHA-256 is `6397b43bcecea024ff68b03a2fcedbf81d726f98fa8dbecb7b08187097dfb9aa`. The official base `model.safetensors` SHA-256 is `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`; the complete processor file hashes are in the run manifest. The official pretraining data overlap with LIBERO is not fully known, so the report keeps that as a limitation. Dataset caches, Python environments, credentials, and large temporary downloads are excluded from the final public archive.
