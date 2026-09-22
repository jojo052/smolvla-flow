# S1 RTC real-time benchmark

## Split-process restart, 2026-09-18

After explicit approval, a new complete run is launched at `/root/autodl-tmp/outputs/s1_rtc_split_formal_v1`, screen `s1-rtc-split-v1`, log `/root/autodl-tmp/logs/s1_rtc_split_formal_v1.log`. At launch, 15 local and remote tests passed. GPU preflight/check/formal completion must be read from this directory, not inferred from launch.

`scripts/run_s1_rtc_split.py` invokes check then run, stopping on any error. It accepts the same output/checkpoint/student/assets-dir/baseline-manifest/protocol arguments as the previous queue. `evaluate_s1_rtc.py --split-process` runs one policy subprocess with four Torch threads and a single controller/environment process with one Torch thread. Pipe RPC has sequential request IDs, bounded response timeouts and propagated worker exceptions. Its payload is the original observation; controller queue timing, actual-age expiry, video rules, policy randomness and all RTC parameters are unchanged. Native request, full RPC round-trip and IPC/dispatch overhead are logged separately. Initial delay uses full round-trip time. No other GPU experiments are launched concurrently.

The old v3 149 completed rows remain separate; its original entry/core/policy/report files were copied into `source_snapshot/` before deployment. They are not counted in the new run. The new manifest records `execution_runtime.version=split-process-v1` and hashes its RPC source. Rejected responses now preserve timing before the admission gate raises. Policy-worker memory peaks are recorded separately from controller resource sampling.

Do not use new code to silently resume the old frozen v3 run. No automatic shutdown or automatic retry is configured.

## Status

2026-09-18: the server is connected. Remote unit tests: 13 passed. The native S1 t=1 guidance gate passed for explicit noise seeds 123 and 124: one native forward, one RTC call, finite nonzero corrections, unchanged parameter digest. This proves path activation only, not a performance benefit. The 24 real-time checks are in progress under `/root/autodl-tmp/outputs/s1_rtc_realtime_v3`. A one-shot `queue_s1_rtc_formal.py` process waits on the check lock and starts formal evaluation only after all 24 checks pass and no error marker exists. There is no retry loop or automatic power control.

Earlier initialization attempts v1/v2 and logs are retained. Startup fixes used the existing offline HF cache and moved controller inspection after lazy environment reset. External/preprocessor images remain 256px; native `policy.prepare_images` applies the checkpoint's internal resize. No model or processor algorithm was changed. Existing checkpoints and historical results are unchanged. Formal completion and backup remain pending.

The locked protocol is `configs/s1_rtc_realtime_v1.json`. No training, dependency upgrades or automatic power control are performed. Formal execution is gated on effective one-step native RTC guidance and all 24 check episodes. If native guidance is zero at the only integration point, stop and report it; do not silently change steps or RTC settings.

## Runtime

Use the existing remote environment and frozen evaluator. Transfer the new entrypoint, policy adapter, report, core module and protocol into the existing project. Do not replace historical result directories. From `/root/autodl-tmp/smolvla-flow`:

```bash
export PYTHONPATH=src:.
export HF_HOME=/root/autodl-tmp/huggingface-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DISTILL40_FROZEN_RUNNER=/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/root/autodl-tmp/venvs/smolvla-flow/bin/python -m scripts.evaluate_s1_rtc \
  --stage check \
  --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero \
  --student /root/autodl-tmp/outputs/distill40_s1_formal_v1/training/010000.pt \
  --assets-dir /root/autodl-tmp/libero-assets \
  --baseline-manifest /root/autodl-tmp/outputs/distill40_s1_formal_v1/formal/manifest.json \
  --protocol configs/s1_rtc_realtime_v1.json \
  --output /root/autodl-tmp/outputs/s1_rtc_realtime_v1
```

After `gate.json` records success for all 24 checks, run the same command with `--stage run`. Interruption recovery uses `--stage resume`. Valid complete episodes, including failures, are skipped. An error marker blocks automatic recovery. Only after inspecting the retained evidence, `--acknowledge-errors` archives the marker and permits a retry; it never deletes the error history.

Report-only execution needs no GPU:

```bash
PYTHONPATH=src:. python3 -m scripts.evaluate_s1_rtc \
  --stage report --output /path/to/copied/s1_rtc_realtime_v1
```

Before shutdown, copy the full experiment directory, including videos, to the local artifact directory and compare a file SHA-256 inventory. Backup verification remains a required operational step, separate from the report's 240-row completeness check.

## Protocol and metrics

One environment, GPU batch 1, absolute 20 Hz deadlines. The waiting action is zero relative motion plus the previous environment gripper command. Native controller `use_delta` evidence is required. Sync discards the old tail when requesting; async retains it; RTC additionally guides using the normalized old tail. No manual blending or gripper hysteresis is introduced.

Admission happens only at control boundaries, with prefix expiry determined from actual observation age. The accepted chunk executes 10 valid actions before another request. Initial chunk prediction precedes the clock. Actual RTC delay >=10 ticks, full expiry, nonfinite output or >5% ticks dispatched more than 5 ms late stops execution. Partial tick/request diagnostics are retained under `errors/`.

Prediction timing includes the native sampling call. Forward timing and RTC processor timing are CUDA-event measurements. RTC extra time subtracts nested native forward time from inclusive RTC time; it includes input-gradient work and is not a separate independent benchmark. Postprocessing and injected waiting are logged separately. Outstanding requests at episode end are recorded as discarded, not accepted.

Episode wall time includes reset, first prediction, control, final request drain and video encoding. Model loading is outside episode time. Resource samples include initialization/check/run processes and are explicitly labelled with that scope. Render work shares the GPU. Chunk jumps are measured only when a new result is accepted, not when a request is submitted.

Report completeness requires 240 distinct formal identities, matching state/student hashes and both preflight gates. It recomputes six condition summaries and paired task-cluster bootstrap intervals. Only eight task clusters are available. Historical six-model performance is not pooled with this changed execution protocol.

## Local verification

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests -p test_rtc_benchmark.py -v
PYTHONPATH=src:. python3 -m scripts.evaluate_s1_rtc --help
```

These tests exercise bookkeeping, queues, deadlines and synthetic report recomputation. They provide no evidence of native GPU/RTC correctness or real 20 Hz timing. Remote preflight is mandatory before formal evaluation.
