import unittest
from smolvla_flow.distill40_sampling import ObservationSampler


def fixture():
    return {'episodes': [{'episode_index': 2*t+i, 'task_index': t,
                          'frames': [['unused', j] for j in range(3)]}
                         for t in range(40) for i in range(2)],
            'splits': {str(t): {'train': [2*t], 'validation': [2*t+1]}
                       for t in range(40)}}


class SamplingTests(unittest.TestCase):
    def test_resume_exact(self):
        a = ObservationSampler(fixture(), 123)
        for _ in range(17): a.sample()
        state = a.state_dict()
        expected = [a.sample() for _ in range(100)]
        b = ObservationSampler(fixture(), 999)
        b.load_state_dict(state)
        self.assertEqual(expected, [b.sample() for _ in range(100)])
        self.assertEqual(a.counts, b.counts)

    def test_no_validation_and_tail_retained(self):
        sampler = ObservationSampler(fixture(), 123)
        samples = [sampler.sample() for _ in range(10000)]
        self.assertEqual(set(t for t, _, _ in samples), set(range(40)))
        self.assertTrue(all(ep % 2 == 0 for _, ep, _ in samples))
        self.assertEqual(set(f for _, _, f in samples), {0, 1, 2})

    def test_leakage_rejected(self):
        m = fixture();m['splits']['0']['validation'] = [0]
        with self.assertRaises(ValueError): ObservationSampler(m, 123)


if __name__ == '__main__': unittest.main()
