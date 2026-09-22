"""Task-balanced observation sampling with serializable deterministic state."""
from collections import Counter
import random


class ObservationSampler:
    def __init__(self, manifest, seed):
        self.episodes = {int(x['episode_index']): x for x in manifest['episodes']}
        self.splits = manifest['splits']
        self.tasks = sorted(int(x) for x in self.splits)
        if len(self.tasks) != 40:
            raise ValueError('Expected all forty tasks')
        for task in self.tasks:
            split = self.splits[str(task)]
            if not split['train'] or not split['validation']:
                raise ValueError('Empty episode split')
            if set(split['train']) & set(split['validation']):
                raise ValueError('Episode leakage')
            for ep in split['train'] + split['validation']:
                if self.episodes[ep]['task_index'] != task:
                    raise ValueError('Episode task mismatch')
        self.rng = random.Random(seed)
        self.counts = Counter()

    def sample(self):
        task = self.rng.choice(self.tasks)
        ep = self.rng.choice(self.splits[str(task)]['train'])
        frame = self.rng.randrange(len(self.episodes[ep]['frames']))
        self.counts[task] += 1
        return task, ep, frame

    def state_dict(self):
        return {'rng': self.rng.getstate(), 'counts': dict(self.counts)}

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])
        self.counts = Counter(state['counts'])
