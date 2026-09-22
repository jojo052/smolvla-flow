# T2→S1 execution record

## Scope

Official `smolvla_libero` initializes both frozen T2 and trainable S1. Only action-expert parameters are updated. No expert-action BC, S5 initialization, RTC, AMP, compilation, or quantization is added. Training consumes exactly 160,000 observations. Historical T10/T5/S5/T2/T1 evaluations remain unchanged.

The 2→1 objective includes the shared initial state in trajectory MSE, so the sum of trajectory and endpoint losses is 1.5 times endpoint MSE over all 50 actions and seven physical action dimensions.

## Implemented independent entries

- `scripts/train_s1.py`: isolated mathematical/recovery preflight, resource benchmark, fixed-budget training segments and 1,280-pair offline validation.
- `scripts/s1_resources.py`: in-process five-second GPU and memory telemetry, no assistant polling.
- `scripts/run_s1_pipeline.py`: sequential train, four closed-loop selections, checkpoint selection, executor consistency checks and final evaluation.
- `scripts/evaluate_s1_interleaved.py`: S1 overlay on the frozen official base, eight independent environments and GPU batch=1.
- `scripts/report_s1.py`: six-model report and task-paired bootstrap comparisons from existing episode records.

## Measured preflight, 2026-09-17

Remote artifacts: `/root/autodl-tmp/outputs/distill40_s1_v3/`.

| Candidate | Samples/s | Input-wait fraction | Mathematical and recovery gate |
|---|---:|---:|---|
| batch16, one decoder | 12.6483 | 58.22% | Passed |
| batch32, one decoder | 13.1308 | 68.48% | Passed |
| batch16, two decoders, resource instrumentation | 13.2960 | 55.52% | Passed |

Batch32 improves sample throughput by about 3.8%, below the predetermined 5% threshold. Batch16 is selected. Because input wait exceeds 20%, the approved two-decoder retest ran separately under `distill40_s1_v4/probe_b16_decode2`. It passed and improved sample throughput by about 5.1% relative to batch16/one decoder, so two decoders are adopted. These results do not establish final closed-loop quality. The projected training-only time is 160,000 / 13.2960 = 3.34 hours; validation and evaluation are additional.

Formal pipeline started in screen `s1-formal-v1` at 2026-09-17 18:50 CST, output `/root/autodl-tmp/outputs/distill40_s1_formal_v1`, log `/root/autodl-tmp/logs/s1_formal_v1.log`. Fresh official initialization, batch16, accumulation1, 10,000 updates, decoder2. The training provenance gate matched and `training.jsonl` was created. Completion is not yet established.

An earlier preflight assertion confused external 256×256 images with internal prepared images. Official preprocessing produces two 512×512 images for this checkpoint. The assertion was corrected to inspect both contracts separately; the model's preprocessing was not changed. Failed probe directories and logs are retained.

Verified in v3: same-batch custom/official Euler output agreement at rtol=1e-5, atol=1e-6; deterministic teacher targets; loss equivalence; checkpoint replay with identical consumed samples and parameter updates within the fixed tolerance; unchanged frozen parameters. CPU regression suite: 21 tests passed, including removing/perturbing expert actions before preprocessing.

## Run after final preflight passes

The chosen gate's source hash must match `train_s1.py`. Do not substitute the v3 gate after updating the entry.

```bash
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/huggingface-cache
export MUJOCO_GL=egl
export DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py
```

Invoke `run_s1_pipeline.py` with `--checkpoint`, `--data`, a new `--output`, `--assets-dir`, the selected `--gate`, and the read-only historical report inputs `--report-artifacts`, `--t1`, `--s5`. All GPU stages are serial. Every 40,000 consumed samples triggers 80 selection episodes, init indices 11/12. The best checkpoint is selected by successes, offline loss, then earlier update. Final evaluation uses 400 episodes, init indices 0–9; four serial and four interleaved pilot episodes must pass first.

The pipeline stops on subprocess errors. Incomplete offline validation or an existing error artifact requires inspection. It does not delete old results, repeat normal failures, increase the budget, or power the instance off.

## Interpretation

Teacher generation is batched during training and may differ numerically from historical single-sample T2. S1 inference uses GPU batch=1. Formal initial states are already known test conditions. T1/S1 parallel efficiency is separate from historical single-environment efficiency. No positive distillation gain is assumed.
