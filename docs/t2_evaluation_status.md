# T2 evaluation dispatch

2026-09-17 10:54:50 Asia/Shanghai: dispatched on AutoDL using existing S5 evaluator without code changes.

- screen: `1500.libero40-t2-direct`
- output: `/root/autodl-tmp/outputs/libero40_t2_direct_v1`
- log: `/root/autodl-tmp/logs/libero40_t2_direct_v1.log`
- evaluator SHA256: `5746d0205d2a9053ed9dbf65c3a527747758ef46d132604d6fe3bd67e2d39df2`
- official checkpoint: `/root/autodl-tmp/checkpoints/smolvla_libero`
- model T2; no student overlay; flow steps 2; chunk 50; execution horizon 1; synchronous; RTC disabled.
- formal initial states 0–9 for each of 40 tasks; 400 episodes; original budgets and seeds preserved.
- frozen runner: `/root/autodl-tmp/benchmarks/libero40_public_v1/scripts/run_libero40.py`
- no new acceleration changes, no automatic shutdown configured, no new monitoring automation.

Command run inside the existing remote environment with `PYTHONPATH=src:.`, `MUJOCO_GL=egl`, offline Hugging Face settings and `DISTILL40_FROZEN_RUNNER`:

```sh
/root/autodl-tmp/venvs/smolvla-flow/bin/python -u scripts/evaluate_distill40.py --checkpoint /root/autodl-tmp/checkpoints/smolvla_libero --model T2 --assets-dir /root/autodl-tmp/libero-assets --output /root/autodl-tmp/outputs/libero40_t2_direct_v1 --phase formal
```

Dispatch was confirmed by screen listing. Completion must be verified from episode files and result.json; screen existence alone does not prove successful evaluation.
