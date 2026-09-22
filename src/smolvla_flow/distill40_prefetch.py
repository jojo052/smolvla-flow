"""Two-batch CPU prefetch with a private lookahead sampler.

Only consumption advances the authoritative sampler saved by save_recovery.
Pending batches are reproducible from that state and can safely be discarded.
One worker owns the reader/cache; it never calls CUDA or a global RNG.
"""
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from .distill40_observations import decode_observation


class ObservationPrefetch:
    def __init__(self, sampler, reader, batch_size=16, depth=2):
        if batch_size < 1 or depth < 1:
            raise ValueError('Positive batch size and depth required')
        self.sampler = sampler
        self.lookahead = copy.deepcopy(sampler)
        self.reader = reader
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='observation-cpu')
        self.pending = deque()
        self.closed = False
        for _ in range(depth):
            self._submit()

    def _prepare(self, requests):
        rows = [decode_observation(self.reader.read(*key)) for key in requests]
        return rows, self.reader.stats()

    def _submit(self):
        requests = [self.lookahead.sample() for _ in range(self.batch_size)]
        self.pending.append((requests, self.pool.submit(self._prepare, requests)))

    def next(self):
        if self.closed:
            raise RuntimeError('Prefetch is closed')
        requests, future = self.pending.popleft()
        rows, stats = future.result()  # Failures propagate without consuming samples.
        actual = [self.sampler.sample() for _ in range(self.batch_size)]
        if actual != requests:
            raise RuntimeError('Prefetch sample order mismatch')
        self._submit()
        return rows, requests, stats

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
