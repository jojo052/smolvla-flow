"""Bounded parquet observation cache; expert action columns are never loaded."""
from collections import OrderedDict
from pathlib import Path


def collate_processed(items):
    import torch
    result = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            if key in ('observation.language.tokens','observation.language.attention_mask'):
                length = max(v.shape[1] for v in values)
                values = [torch.nn.functional.pad(v,(0,length-v.shape[1]),value=0) for v in values]
            if any(v.shape[1:] != values[0].shape[1:] for v in values):
                raise ValueError('Unmatched processed shapes: '+key)
            result[key] = torch.cat(values,dim=0)
    return result


class ObservationReader:
    columns = ('observation.images.image', 'observation.images.image2', 'observation.state')

    def __init__(self, root, manifest, max_cached_files=4, max_cache_bytes=None):
        if max_cached_files < 1:
            raise ValueError('Cache must hold at least one file')
        self.root = Path(root)
        self.episodes = {int(e['episode_index']): e for e in manifest['episodes']}
        self.splits = manifest['splits']
        self.cache = OrderedDict()
        self.max_cached_files = max_cached_files
        if max_cache_bytes is not None and max_cache_bytes <= 0:
            raise ValueError('Cache byte budget must be positive')
        self.max_cache_bytes = max_cache_bytes
        self.cache_bytes = 0
        self.hits = self.misses = 0

    def read(self, task, episode, frame):
        import pyarrow.parquet as pq
        e = self.episodes[episode]
        if e['task_index'] != task:
            raise ValueError('Task/episode mismatch')
        relative, offset = e['frames'][frame]
        if relative not in self.cache:
            self.misses += 1
            table = pq.read_table(self.root/relative, columns=list(self.columns))
            size = table.get_total_buffer_size()
            while self.cache and (self.cache_bytes + size > self.max_cache_bytes if self.max_cache_bytes is not None else len(self.cache) >= self.max_cached_files):
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.get_total_buffer_size()
            if self.max_cache_bytes is not None and size > self.max_cache_bytes:
                row = table.slice(offset,1).to_pylist()[0]
                row['task'] = self.splits[str(task)]['language']
                return row
            self.cache[relative] = table
            self.cache_bytes += size
        else:
            self.hits += 1
        self.cache.move_to_end(relative)
        row = self.cache[relative].slice(offset,1).to_pylist()[0]
        row['task'] = self.splits[str(task)]['language']
        return row

    def stats(self):
        total = self.hits + self.misses
        return dict(cache_hits=self.hits, cache_misses=self.misses,
                    cache_hit_rate=self.hits/total if total else 0., cache_bytes=self.cache_bytes)


def decode_observation(row):
    """CPU-only image decode. No processor, CUDA, or random-number calls."""
    import io
    import numpy as np
    from PIL import Image
    result = dict(row)
    for key in ObservationReader.columns[:2]:
        value = row[key]
        if not isinstance(value,dict) or value.get('bytes') is None:
            raise ValueError('Expected embedded image bytes')
        image = np.array(Image.open(io.BytesIO(value['bytes'])).convert('RGB'),copy=True)
        if image.shape != (256,256,3):
            raise ValueError('Expected 256x256 RGB image')
        result[key] = image
    return result


def processed_observation(row, preprocessor, *, decoded=False):
    import io
    import numpy as np
    import torch
    from PIL import Image
    raw = {'task': row['task'], 'observation.state': torch.tensor(row['observation.state'],dtype=torch.float32)}
    if raw['observation.state'].shape != (8,):
        raise ValueError('Expected eight state dimensions')
    if not decoded:
        row = decode_observation(row)
    for key in ObservationReader.columns[:2]:
        value = row[key]
        image = value
        if image.shape != (256,256,3):
            raise ValueError('Expected 256x256 RGB image')
        raw[key] = torch.from_numpy(image).permute(2,0,1).float()/255.0
    return preprocessor(raw)
