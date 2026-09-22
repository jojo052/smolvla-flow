# S5 batch16 CPU prefetch gate

The optional `--cpu-prefetch` path uses one reader with a 16 GiB Arrow-buffer
budget and one CPU worker preparing two batches of 16 observations. Image
decoding alone moves to the worker. Processor calls and CUDA remain on the
training thread. Existing default flags and experiment budgets are unchanged.

The worker samples through a private cloned sampler. The checkpoint sampler
advances only when a batch is consumed. Pending lookahead is discarded before
validation and can be regenerated after restore. A failed preparation propagates
without consuming the authoritative sampler. No expert action column is loaded.

## Completed checks

- Two unit tests: ordered consumption/pending recovery and failure propagation.
- Actual data CPU check: 40 consumed batches, 640 samples, matched baseline order;
  first sample both images matched pixel for pixel.
- Including worker lookahead: 287 hits, 369 misses, cold hit rate 43.75%.
- Table cache: 17,143,970,813 bytes. Process peak RSS: 19,574,484 KiB.
- CPU-only check elapsed 64.29 seconds while formal GPU training continued.
  This is not a training speed measurement.

## Pending checkpoint-gated GPU test

Supervisor waits for batch16 stage1 `001000.pt` plus matching 1280-pair offline
validation with 40 tasks, 32 pairs each and finite losses. It verifies the
identities of trainer/pipeline/evaluation queue, stops the old computation, and
tests original versus prefetch from that recovery point. All probe updates are
isolated. Formal resumption starts from the original checkpoint.

Gate fixed before measurement: identical sample sequences and first-batch input
and noise hashes, 20-update parameter comparison rtol 1e-5 / atol 1e-7, frozen
teacher and backbone hashes unchanged, save-with-lookahead recovery equivalence,
finite updates, process peak RSS below 60 GiB and measured speedup above 1.05.
Two warmup updates excluded; 18 timed updates per mode. Input hashing only in
the excluded first update and the separate recovery check. Short benchmark
does not promise sustained speed or final policy success.

On gate failure/timeout the supervisor resumes the original batch16 path.
On pass it appends only `--cpu-prefetch`, preserving batch16, learning rate,
teacher, total updates and evaluation schedule. It reattaches the existing
formal S5 evaluation command to the new pipeline PID. It never launches S2.

Remote evidence directory:
`/root/autodl-tmp/outputs/distill40_s5_b16_v1/prefetch_probe`

The checkpoint gate checks every 15 seconds inside the server, without model
calls. Steps computed between checkpoint validation and stopping may be replayed
from update 1000; no saved result is deleted or overwritten by the probe.
