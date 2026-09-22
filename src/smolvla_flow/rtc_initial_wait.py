"""Initial waiting precedes the control clock; runtime admission is unchanged."""
import math
from collections import deque
from smolvla_flow.rtc_benchmark import Timeline, checked_chunk


class InitialWaitTimeline(Timeline):
    def prime(self, normalized, actions, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid first-chunk duration')
        self.queue = deque(zip(checked_chunk(normalized), checked_chunk(actions)))
        # Preserve the measured estimate even if it is >= the guiding horizon.
        # No environment ticks elapse during this initial prediction.
        self.last_delay = math.ceil(elapsed * 20)
