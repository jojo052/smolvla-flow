# RTC latency diagnosis, 2026-09-18

Formal v3 remains paused at 149/240. This investigation did not resume episodes, relax the 10-tick horizon, change weights, or overwrite results.

## Fixed-input experiment

Remote output: `/root/autodl-tmp/outputs/s1_rtc_latency_probe_v1/diagnosis.json`.
Local copy: `artifacts/s1_rtc_latency_probe_v1/diagnosis.json`.
Entry: `scripts/probe_s1_rtc_latency.py`.

One S1 model, fixed Long task5/init13 observation, identical explicit noise, 4 PyTorch threads. Each mode/condition has 3 warmup predictions and 15 measured predictions. No artificial latency is injected. Concurrent environment uses zero relative motion with fixed gripper; this is diagnostic load, not a closed-loop success evaluation.

| Condition | No RTC request P50/P95 ms | RTC request P50/P95 ms |
|---|---:|---:|
| Main thread, environment static | 106.1 / 109.8 | 105.9 / 109.7 |
| Worker thread, environment static | 100.4 / 102.1 | 101.5 / 107.4 |
| Worker thread, environment stepping/rendering at 20 Hz | 236.7 / 258.6 | 201.3 / 251.7 |

Request timing here includes preprocessing, native prediction, postprocessing and return bookkeeping. All output comparisons passed rtol=1e-5, atol=1e-6. This is evidence for the tested fixed input only. RTC extra CUDA-event time under render load has P50 1.34 ms and P95 2.47 ms. The difference between the two render-mode medians is not evidence that RTC improves latency; this is a small sequential timing probe.

Concurrent environment execution causes substantial slowdown in this setup. Thread dispatch alone does not reproduce it. CPU/GIL, GPU rendering contention, and synchronization have not been independently isolated. Do not label the wall-time delta as pure GPU compute.

## Original failure evidence

Long task5/init20/RTC+100ms failed at attempted admission tick93, with the next request inferred from the fixed scheduler to have been submitted at tick83. The preceding accepted request: preprocessing10.23ms, sampling296.51ms, postprocessing19.52ms, injected100.08ms, total426.33ms, accepted after9ticks. The preceding three accepted requests each took9ticks. No waiting actions were needed; recorded tick lateness max1.18ms, zero ticks exceeding5ms.

The failed request's own timing was not retained because admission raised before metrics were appended. Local `accept_recorded` now saves the result before admission and retains `accepted_tick=null` on rejection. A regression test covers this; 14 unit tests pass. The modified formal entry/core have deliberately not replaced the locked remote v3 sources. Any future deployment must explicitly preserve provenance, rather than silently change v3's frozen manifest.

Suggested next experiment: separate environment and policy processes while preserving one environment, 20Hz, GPU batch1, images and noise. Repeat the same fixed-input comparison, then timing pilots. This may remove same-process contention but cannot guarantee relief from shared-GPU rendering contention. A material runtime change requires a separately versioned benchmark, not pooling old/new latency or success records without disclosure.

## Completed split-process probe

Entry now supports `--split-process`; the original probe mode remains available. Results are in `artifacts/s1_rtc_latency_split_v1/diagnosis.json`, copied from the same-named remote output directory. A spawned process owns the single environment and EGL context, and uses one PyTorch CPU thread. The parent retains the only policy model with four PyTorch threads. Each condition has three warmups and fifteen measured predictions, no injected waiting.

| Concurrent environment architecture | No RTC request P50/P95 ms | RTC request P50/P95 ms |
|---|---:|---:|
| Same process, prior probe | 236.7 / 258.6 | 201.3 / 251.7 |
| Separate process, new probe | 104.0 / 109.3 | 105.1 / 113.4 |

The spawned environment's initial observation exactly matches the parent reference. All fixed-input action comparisons passed rtol=1e-5, atol=1e-6. Static-worker controls in the new run were105.0ms without RTC and104.6ms with RTC, close to their respective concurrent values. No GPU or model parallelism was introduced.

This supports process isolation as an engineering remedy for the measured concurrency slowdown. It does not identify GIL versus driver/context contention individually, and a zero-motion short probe does not establish worst-case timing in a moving closed loop. Request timings do not include a production observation/action IPC pipeline: the parent uses its fixed observation and the child generates rendering load. That IPC path and 24 timing pilots still require implementation/verification before formal use. Neither the old149 rows nor historical model results were modified or resumed. A new240-episode run would incur additional compute and requires explicit approval rather than merging two runtime versions.
