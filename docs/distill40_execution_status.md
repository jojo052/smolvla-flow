# Four-suite progressive distillation execution

## Shared-prefix recovery and S5 queue

The shared-prefix probe matched loss, gradient norm and updated parameters
exactly for its matched update. Real-model save/restore continuation also had
zero maximum parameter error. Microbatch 2 and 4 failed the predeclared gate
despite matching input hashes, so neither was adopted.

The resumed stage-one process was observed as PID 19705 with parent pipeline
19703. Its execution journal confirms update 1000 and the original checkpoint
hash, shared_prefix=true, microbatch=1, accumulation=8. Old source files are
backed up remotely under acceleration/*.before_shared.py. The queue process
19897 waits for that identified pipeline and runs formal S5 only after selection
and selected-checkpoint hash verification. Runtime PIDs are historical evidence;
check them before signaling. Formal output is distill40_v1/formal_S5.

The report entry now accepts --s5-only without changing the default four-group
requirement. A synthetic 800-episode regression covers the two-group output;
it does not represent measured policy outcomes.

## Latest execution scope and acceleration checkpoint (2026-09-15)

The user has deferred S2. Current delivery is selected S5, its 400 formal
episodes, and comparison with the frozen T10 baseline. The original four-model
scope below remains a historical plan, not authorization to start stage two.

Stage-one training has started. PID 14018 was verified in stopped state `Tl+`
during acceleration preparation. Resume must use the verified 001000 recovery
checkpoint, retain effective batch eight and the 20,000-update budget, and
preserve scheduled selection. Do not infer current process state from this file.

An isolated CPU test passed microbatch mean weighting (1, 2, 4, 8) and exact
save/restore continuation. These are small-model tests, not proof of unchanged
SmolVLA success rates. `scripts/probe_distill40_microbatch.py` now compares real
model updates using matched observations and noise. Its first attempt stalled
on an online configuration lookup and was interrupted before training. The
replacement uses the formal process's HF_HOME and HF_HUB_OFFLINE settings.
Results are remote `acceleration/microbatch_probe.json` under distill40_v1.
No accelerated configuration has been adopted yet. Single-trial compute timing
excludes preprocessing and must not be used as a complete training ETA.

The sections below describe the earlier preparation snapshot and are retained
for provenance; their pending gates and "not started" text are historical.

## Fixed scope

T10 official frozen teacher, unchanged-weight T2, distilled S5, distilled S2.
Transitions 10->5->2 use the existing trajectory and teacher endpoint losses.
No expert action loss, RTC, ACT training, or task34 LoRA teacher.
Each stage: 20,000 optimizer updates, microbatch 1, accumulation 8.
Closed-loop selection: updates 5000/10000/15000/20000, all forty tasks,
initial states 11 and 12. Formal initial states 0 through 9 are not selection data.

## Current evidence

- Data disk confirmed 150 GiB with approximately 119 GiB free before download.
- Dataset revision: `86958911c0f959db2bbbdb107eb3e17c5f9c798e`.
- Forty source task instructions exactly match the frozen benchmark manifest.
- Source metadata declares 1693 episodes and 273465 frames.
- Source episode file offsets conflict with actual parquet footers, including
  file 000 and file 068. Historical metadata also contains inconsistent offsets.
- Full footer audit is in progress under remote
  `/root/autodl-tmp/outputs/distill40_v1/data_footer_audit`.
- `prepare_distill40_data.py` requires contiguous footer coverage before download,
  then verifies hashes and derives observation membership from actual row indices.
  Original source metadata remains unchanged. Episode lengths and instructions
  must agree with actual rows before a derived manifest is published.
- Three sampler CPU tests pass. A tiny torch model recovery continuation test
  passes exactly on CPU. These do not establish real SmolVLA preflight success.

## Remaining gates

1. Complete footer audit and download/verify all data files.
2. Verify actual rows, publish episode splits and fixed validation observations.
3. Finish multi-task training and independent evaluation entrypoints.
4. Run real-model strict load, Euler equivalence, freeze, teacher determinism,
   action-independence, and recovery tests. Measure GPU cost for both stages.
5. Run both stages and scheduled offline/closed-loop validation with fixed selection.
6. Freeze selected students; evaluate T2/S5/S2 and verify reused T10 protocol.
7. Generate all required paired statistics, latency tables, provenance, and videos.

Training has not started. Never infer completion from this status file; inspect
live processes and emitted results. Existing scripts and checkpoints are retained.
