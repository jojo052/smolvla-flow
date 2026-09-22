"""Fixed-input strength probes. No closed-loop runs or old artifact changes."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for key in ('source', 'output', 'checkpoint', 'student'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--weights', type=float, nargs='+', default=[1.25, 1.5, 1.75])
    args = p.parse_args()
    import numpy as np
    import torch
    from scripts.s1_rtc_policy import Policy
    from scripts.evaluate_s1_rtc import STUDENT_SHA
    from smolvla_flow.benchmark40 import atomic_json, sha256
    summary = json.loads((args.source / 'summary.json').read_text())
    for name, key in [('observations.pt', 'observation_sha256'), ('noise_prefix.pt', 'noise_prefix_sha256')]:
        assert sha256(args.source / name) == summary[key], name
    assert sha256(args.student) == STUDENT_SHA
    args.output.mkdir(exist_ok=False, parents=True)
    # Trusted tensors produced by our prior isolated diagnostic.
    samples = torch.load(args.source / 'observations.pt', weights_only=False, map_location='cpu')
    stored = torch.load(args.source / 'noise_prefix.pt', weights_only=False, map_location='cpu')
    torch.set_num_threads(4)
    policy = Policy(args)
    before = policy.digest()
    weights = list(dict.fromkeys([0., 1.] + args.weights))
    atomic_json(args.output / 'design.json', dict(weights=weights, delays=[3, 5],
        source_summary_sha256=sha256(args.source / 'summary.json'), student_sha256=STUDENT_SHA,
        script_sha256=sha256(Path(__file__)), rtol=1e-5, atol=1e-6,
        purpose='fixed-input diagnostic only; old full-block gates retained',
        caveat='executed slice uses estimated delays 3/5, not measured closed-loop observation age'))
    rows = []
    for si, sample in enumerate(samples):
        for seed in (123, 124):
            item = stored[si * 2 + seed - 123]
            prefix = item['prefix'][0].numpy()
            for delay in (3, 5):
                mask = policy.rtc.get_prefix_weights(delay, 10, 50)[:40].cpu().numpy()[:, None]
                plain = None
                for weight in weights:
                    policy.rtc.rtc_config.max_guidance_weight = weight
                    result = policy.predict(sample['observation'], sample['task']['language'],
                        'rtc' if weight else 'async', 0, prefix.tolist() if weight else None,
                        delay, item['noise'].cuda().clone(), diagnostic=True)
                    out = np.asarray(result['normalized'], dtype=np.float32)
                    assert result['metrics']['forward_calls'] == 1
                    assert result['metrics']['timesteps'] == [[1.0]]
                    if weight in (0., 1.):
                        old = np.load(args.source / f'S1_{si}_{seed}_{delay}_{weight}.npy')[0]
                        np.testing.assert_allclose(out, old, rtol=1e-5, atol=1e-6)
                    if weight == 0:
                        plain = out.copy()
                    envelope = max(float(np.abs(plain).max()), float(np.abs(prefix).max()))
                    error = float(((out[:40] - prefix)**2 * mask).mean())
                    base_error = float(((plain[:40] - prefix)**2 * mask).mean())
                    sl = slice(delay, 10)
                    executed_error = float(((out[sl] - prefix[sl])**2 * mask[sl]).mean())
                    executed_base = float(((plain[sl] - prefix[sl])**2 * mask[sl]).mean())
                    lower = np.minimum(plain[:40], prefix)
                    upper = np.maximum(plain[:40], prefix)
                    excess = np.maximum(lower - out[:40], out[:40] - upper)
                    actions = np.asarray(result['actions'])
                    row = dict(sample=si, suite=sample['task']['suite'], tick=sample['tick'],
                        seed=seed, delay=delay, weight=weight, weighted_error=error, plain_error=base_error,
                        executed_error=executed_error, executed_base_error=executed_base,
                        normalized_max=float(np.abs(out).max()), envelope=envelope,
                        full_gate=error <= base_error + 1e-6 and np.abs(out).max() <= envelope + 1e-5,
                        full_envelope_pass=bool(np.abs(out).max() <= envelope + 1e-5),
                        executed_coordinate_excess=float(max(0., excess[sl].max())),
                        executed_motion_over1=float((np.abs(actions[delay:, :6]) > 1).any(axis=1).mean()),
                        first_usable_gain=float(weight * mask[delay, 0]), metrics=result['metrics'])
                    row['full_gate'] = bool(row['full_gate'])
                    rows.append(row)
                    np.save(args.output / f'S1_{si}_{seed}_{delay}_{weight}.npy', out)
        atomic_json(args.output / 'partial.json', dict(rows=rows))
        print('DONE', si + 1, len(samples), flush=True)
    assert policy.digest() == before
    aggregates = []
    for weight in weights:
        rr = [r for r in rows if r['weight'] == weight]
        aggregates.append(dict(weight=weight, cases=len(rr), full_gate_pass=sum(r['full_gate'] for r in rr),
            weighted_error_ratio=sum(r['weighted_error'] for r in rr)/sum(r['plain_error'] for r in rr),
            executed_error_ratio=sum(r['executed_error'] for r in rr)/sum(r['executed_base_error'] for r in rr),
            executed_overshoot_cases=sum(r['executed_coordinate_excess'] > 1e-5 for r in rr),
            max_executed_excess=max(r['executed_coordinate_excess'] for r in rr),
            motion_over1_mean=sum(r['executed_motion_over1'] for r in rr)/len(rr)))
    atomic_json(args.output / 'summary.json', dict(rows=rows, aggregates=aggregates,
        parameter_sha256=before, reference_reproduction_passed=True, parameters_unchanged=True))
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == '__main__':
    main()
