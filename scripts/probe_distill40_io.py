"""Read-only CPU input timing from the formal checkpoint's sampler state."""
import argparse
import json
import time
from pathlib import Path
import torch
from smolvla_flow.distill40_sampling import ObservationSampler
from smolvla_flow.distill40_observations import ObservationReader, processed_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data/'observation_manifest.json').read_text())
    state = torch.load(args.recovery, map_location='cpu', weights_only=False)
    sampler = ObservationSampler(manifest, 123)
    sampler.load_state_dict(state['sampler'])
    samples = [sampler.sample() for _ in range(80)]
    del state
    results = []
    for capacity in (4, 16):
        reader = ObservationReader(args.data, manifest, max_cached_files=capacity)
        read_seconds = decode_seconds = 0.0
        for sample in samples:
            start = time.perf_counter()
            row = reader.read(*sample)
            read_seconds += time.perf_counter()-start
            start = time.perf_counter()
            processed_observation(row, lambda raw: raw)
            decode_seconds += time.perf_counter()-start
        results.append(dict(cache_files=capacity, samples=len(samples),
                            read_seconds=read_seconds, decode_seconds=decode_seconds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'results': results, 'note': 'CPU only; sequential runs may benefit from OS page cache'}, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
