# Historical task34 appendix

This appendix preserves the original single-task experiment. It is not part of the primary 40-task comparison.

## Protocol

- LIBERO-Spatial task 0, corresponding to dataset task index 34
- 30 environment seeds, 0-29, per strategy
- Policy seed 123
- Two 256x256 RGB observations and an 8D robot state
- 7D action; each chunk executes 10 actions
- 280-step episode limit
- Synchronous execution with RTC disabled
- Success is LIBERO `info["is_success"]`

ACT and Diffusion were trained on task34-only data. The official SmolVLA checkpoint was not given an equivalent task34 training budget. The results are therefore checkpoint-level deployment evidence, not a fair equal-training comparison across model families.

## Results

| Strategy | Success | Wilson 95% interval | Complete rollout throughput | Mixed `select_action` mean |
| --- | ---: | ---: | ---: | ---: |
| ACT 050000 | **29/30 (96.67%)** | 83.33%-99.41% | **19.99 tick/s** | **1.76 ms** |
| Diffusion 080000 | 25/30 (83.33%) | 66.44%-92.66% | 12.26 tick/s | 40.93 ms |
| Official SmolVLA, 10 steps | 22/30 (73.33%) | 55.55%-85.82% | 13.32 tick/s | 36.26 ms |
| Direct SmolVLA, 2 steps | 21/30 (70.00%) | 52.12%-83.34% | 19.31 tick/s | 15.04 ms |
| Development distilled SmolVLA, 2 steps | 20/30 (66.67%) | 48.78%-80.77% | 19.27 tick/s | 14.97 ms |

The full historical raw results remain under `artifacts/rollout/` and `artifacts/baselines/`. This appendix is kept so that the original task34 evidence remains auditable, while the README and final report use the 40-task benchmark as the main result.
