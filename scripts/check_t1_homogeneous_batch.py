"""Isolate batch arithmetic from language padding using identical inputs."""
import argparse
from collections import Counter
from pathlib import Path
import torch
from scripts.probe_libero40_t1_parallel import stack, compare
from scripts.run_libero_rollout import _load_policy, parse_args
from smolvla_flow.benchmark40 import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--probe', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    loaded = _load_policy(parse_args(['--checkpoint', args.checkpoint, '--flow-steps', '1',
                                     '--disable-rtc']), action_execution_steps=1)
    policy = loaded.policy.eval()
    bank = torch.load(args.probe/'inputs.pt', map_location='cpu', weights_only=False)
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
             for k, v in bank['observations'][0].items()}
    noise = bank['noise'][0].cuda()
    report = {'parameter_dtypes': dict(Counter(str(p.dtype) for p in policy.parameters())),
              'results': [], 'formal_authorized': False}
    # Confirm explicit per-episode generator matches the original global stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(123)
        native = [policy.model.sample_noise((1, 50, 32), 'cuda') for _ in range(8)]
    generator = torch.Generator(device='cuda').manual_seed(123)
    explicit = [torch.normal(0., 1., size=(1, 50, 32), device='cuda', generator=generator)
                for _ in range(8)]
    report['native_noise_sequence_equal'] = all(torch.equal(a,b) for a,b in zip(native, explicit))
    with torch.inference_mode():
        policy.reset()
        reference = policy._get_action_chunk(batch, noise=noise.clone())
        for size in (1, 2, 4, 8, 16, 32):
            actual = policy._get_action_chunk(stack([batch]*size), noise=noise.repeat(size, 1, 1))
            item = {'batch': size, **compare(actual, reference.expand(size, -1, -1)),
                    'first_action': compare(actual[:, 0], reference[:, 0].expand(size, -1))}
            report['results'].append(item)
            print(item, flush=True)
    atomic_json(args.probe/'homogeneous_batch_gate.json', report)


if __name__ == '__main__':
    main()
