import unittest
from smolvla_flow.rtc_benchmark import Timeline, accept_recorded
from smolvla_flow.rtc_initial_wait import InitialWaitTimeline


def chunk():
    return [[float(i)] * 7 for i in range(50)]


class InitialWaitTests(unittest.TestCase):
    def test_old_gate_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'Initial RTC'):
            Timeline('rtc').prime(chunk(), chunk(), .55)

    def test_first_chunk_does_not_expire_and_estimate_is_uncapped(self):
        for mode in ('sync', 'async', 'rtc'):
            q = InitialWaitTimeline(mode)
            q.prime(chunk(), chunk(), .55)
            self.assertEqual(q.last_delay, 11)
            self.assertEqual(len(q.queue), 50)
            a, waiting, age = q.pop(0)
            self.assertEqual(a, tuple(chunk()[0]))
            self.assertEqual((waiting, age), (False, 0))
            for tick in range(1, 10):
                q.pop(tick)
            _, _, estimate = q.submit(10)
            self.assertEqual(estimate, 11)

    def test_runtime_ten_tick_gate_and_record_survive(self):
        q = InitialWaitTimeline('rtc')
        q.prime(chunk(), chunk(), .55)
        for tick in range(10):
            q.pop(tick)
        rid, _, _ = q.submit(10)
        records = []
        with self.assertRaisesRegex(ValueError, 'RTC delay exceeds'):
            accept_recorded(q, records, rid, 20,
                dict(normalized=chunk(), actions=chunk(), metrics=dict(elapsed_s=.5)))
        self.assertEqual(records[0]['actual_delay'], 10)
        self.assertIsNone(records[0]['accepted_tick'])

    def test_runtime_nine_ticks_still_crops(self):
        q = InitialWaitTimeline('rtc')
        q.prime(chunk(), chunk(), .55)
        for tick in range(10):
            q.pop(tick)
        rid, _, _ = q.submit(10)
        self.assertEqual(q.accept(rid, 19, chunk(), chunk()), 9)
        self.assertEqual(q.pop(19)[0], tuple(chunk()[9]))

    def test_same_behavior_below_initial_gate(self):
        for mode in ('sync', 'async', 'rtc'):
            old, new = Timeline(mode), InitialWaitTimeline(mode)
            for q in (old, new):
                q.prime(chunk(), chunk(), .34)
            self.assertEqual(old.__dict__, new.__dict__)


if __name__ == '__main__':
    unittest.main()
