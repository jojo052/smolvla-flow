# LIBERO task34 formal rollout results

This directory records the completed five-policy comparison for LIBERO-Spatial
task 0, which maps to dataset task index 34.

## Locked evaluation contract

- Environment seeds: `0..29`
- Policy sampling seed: `123`
- Episodes per policy: `30`
- Episode limit: `280` simulator steps
- Executed action prefix: `10` actions per generated chunk
- Simulator control semantics: `20 Hz`
- Cameras: `agentview_image`, `robot0_eye_in_hand_image`
- Observation resolution: `256x256`
- State and action dimensions: `8` and `7`
- Gripper polarity: `positive_open`
- Runtime mode: synchronous, RTC disabled
- Success condition: LIBERO terminal `info["is_success"]`

All five result files completed the exact seed set, used policy seed 123, and
recorded only finite actions. Every failed episode reached the 280-step limit.

## Results

| Policy | Success | Wilson 95% interval | Mean rollout throughput | Mixed `select_action` mean |
| --- | ---: | ---: | ---: | ---: |
| ACT checkpoint 050000 | 29/30 (96.67%) | 83.33% to 99.41% | 19.99 tick/s | 1.76 ms |
| Diffusion checkpoint 080000 | 25/30 (83.33%) | 66.44% to 92.66% | 12.26 tick/s | 40.93 ms |
| Native SmolVLA, 10 Flow steps | 22/30 (73.33%) | 55.55% to 85.82% | 13.32 tick/s | 36.26 ms |
| Undistilled SmolVLA, 2 Flow steps | 21/30 (70.00%) | 52.12% to 83.34% | 19.31 tick/s | 15.04 ms |
| Development distilled SmolVLA, 2 Flow steps | 20/30 (66.67%) | 48.78% to 80.77% | 19.27 tick/s | 14.97 ms |

`mixed_select_action_call_mean_seconds` combines full chunk-generation calls
with inexpensive action-queue pop calls. It describes the measured policy-call
cost per simulator tick under this execution contract. It is not a standalone
full-forward latency benchmark.

## Paired statistics

Every policy used the same environment seeds. Pairwise success comparisons use
the two-sided exact McNemar test. The four predeclared ACT comparisons use the
Holm step-down correction.

| Comparison | Discordant wins | Exact two-sided p | Holm-adjusted p |
| --- | ---: | ---: | ---: |
| ACT vs Diffusion | 4 to 0 | 0.125000 | 0.125000 |
| ACT vs native SmolVLA | 7 to 0 | 0.015625 | 0.031250 |
| ACT vs undistilled 2-step SmolVLA | 8 to 0 | 0.0078125 | 0.0234375 |
| ACT vs development distilled 2-step SmolVLA | 9 to 0 | 0.00390625 | 0.015625 |

The exact McNemar value for discordant counts `b` and `c` is the two-sided
binomial probability under `Binomial(b + c, 0.5)`, capped at one. Holm-adjusted
values are calculated over the four ACT comparisons shown above.

## Checkpoint selection and interpretation limits

ACT uses the task34-trained checkpoint at step 50000. Diffusion training
completed through step 100000; checkpoint 080000 was selected before rollout
because it had the lowest held-out action MSE among checkpoints 020000 through
100000. The corresponding validation artifacts are stored under
`artifacts/baselines/task34_real_v2/`.

The SmolVLA base is the official `HuggingFaceVLA/smolvla_libero` snapshot
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. ACT and Diffusion received
task34-only training, while the native SmolVLA checkpoint did not receive an
equivalent task34 training budget. The comparison therefore supports a
checkpoint-level deployment decision for this task. It does not isolate pure
architecture quality under matched pretraining and optimization budgets.

The distilled adapter is a development artifact from `task0_dev5_final`. It
used 8 samples and 10 optimization updates and has 97,420,192 parameters. Its
result cannot support a completed task34-distillation claim. Under the current
data, reducing native SmolVLA from 10 to 2 Flow steps improved runtime, while
the development adapter did not recover success rate.

## Files

- `comparison_summary.json`: consolidated metrics, provenance, integrity
  checks, exact McNemar tests, and Holm correction
- `native_teacher_10step_sync_seeds0_29.json`: native SmolVLA episodes
- `undistilled_2step_sync_seeds0_29.json`: undistilled 2-step SmolVLA episodes
- `distilled_dev_2step_sync_seeds0_29.json`: development distilled episodes
- `../baselines_task34_real_v2_formal/act_050000_sync_seeds0_29.json`: ACT episodes
- `../baselines_task34_real_v2_formal/diffusion_080000_sync_seeds0_29.json`: Diffusion episodes
