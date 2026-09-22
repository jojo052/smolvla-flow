"""Fixed-budget S1 train/selection/formal pipeline. No old run is changed."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from smolvla_flow.benchmark40 import atomic_json, sha256, freeze_manifest


def choose_probe(reports):
    eligible = [r for r in reports if r.get('passed')]
    if not eligible:
        raise ValueError('No eligible resource/preflight candidate')
    by_batch = {r['provenance']['batch']: r for r in eligible}
    if len(by_batch) != len(eligible):
        raise ValueError('Duplicate candidate batch')
    if 16 not in by_batch:
        return by_batch[32]
    if 32 not in by_batch:
        return by_batch[16]
    return by_batch[32] if by_batch[32]['samples_per_second'] >= 1.05 * by_batch[16]['samples_per_second'] else by_batch[16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'output', 'assets-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True, help='Selected, completed probe.json')
    for name in ('report-artifacts','t1','s5'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if list(args.output.rglob('error.json')):
        raise RuntimeError('Recorded error requires inspection')
    gate = json.loads(args.gate.read_text())
    if not gate['passed']:
        raise ValueError('Preflight failed')
    batch = gate['provenance']['batch']
    if batch not in (16, 32):
        raise ValueError('Unapproved batch')
    source_files = [root/'scripts/train_s1.py', root/'scripts/evaluate_s1_interleaved.py',
                    root/'scripts/s1_resources.py',root/'scripts/report_s1.py',Path(__file__)]
    source_files += sorted((root/'src/smolvla_flow').glob('distill*.py'))
    freeze_manifest(args.output/'pipeline_manifest.json', {
        'gate_sha256': sha256(args.gate), 'sources': {str(p.relative_to(root)): sha256(p) for p in source_files},
        'budget': 160000, 'batch': batch, 'base_sha256': gate['provenance']['base_sha256']})

    def run(script, options):
        subprocess.run([sys.executable, '-u', str(root/'scripts'/script), *map(str, options)], check=True, cwd=root)

    train = args.output/'training'
    train.mkdir(exist_ok=True)
    candidates = []
    for samples in (40000, 80000, 120000, 160000):
        step = samples // batch
        checkpoint = train/f'{step:06d}.pt'
        offline = train/f'offline_{step:06d}.json'
        if not checkpoint.exists():
            saved = sorted(train.glob('[0-9][0-9][0-9][0-9][0-9][0-9].pt'))
            options = ['--checkpoint', args.checkpoint, '--data', args.data, '--output', train,
                       '--mode', 'train', '--batch', batch, '--gate', args.gate,
                       '--decode-workers', gate['provenance']['decode_workers']]
            if saved:
                options += ['--resume', saved[-1]]
            run('train_s1.py', options)
        if not checkpoint.exists() or not offline.exists():
            raise RuntimeError('Checkpoint or complete offline validation missing')
        loss = json.loads(offline.read_text())
        if len(loss['pairs']) != 1280 or loss['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Offline validation provenance mismatch')
        destination = args.output/f'validation_{step:06d}'
        result_path = destination/'validation_run/result.json'
        if not result_path.exists():
            run('evaluate_s1_interleaved.py', ['--checkpoint', args.checkpoint, '--student', checkpoint,
                '--assets-dir', args.assets_dir, '--output', destination, '--mode', 'validation', '--workers', 8])
        result = json.loads(result_path.read_text())
        rows = result['episodes']
        if not result['complete'] or len(rows) != 80 or len({r['key'] for r in rows}) != 80:
            raise ValueError('Incomplete closed-loop selection')
        digest = sha256(checkpoint)
        if any(r['checkpoint_sha256'] != digest or r['init_index'] not in (11, 12) for r in rows):
            raise ValueError('Selection weight/initial-state mismatch')
        candidates.append(dict(update=step, samples=samples, successes=sum(r['success'] for r in rows),
            offline_loss=loss['mean_loss'], checkpoint=str(checkpoint), checkpoint_sha256=digest))
        atomic_json(args.output/'candidates.json', candidates)
    best = min(candidates, key=lambda x: (-x['successes'], x['offline_loss'], x['update']))
    freeze_manifest(args.output/'selection.json', best)
    destination = args.output/'formal'
    if not (destination/'formal_run/result.json').exists():
        mode = 'resume' if (destination/'pilot_gate.json').exists() else 'check-and-run'
        run('evaluate_s1_interleaved.py', ['--checkpoint', args.checkpoint, '--student', best['checkpoint'],
            '--assets-dir', args.assets_dir, '--output', destination, '--mode', mode, '--workers', 8])
    result = json.loads((destination/'formal_run/result.json').read_text())
    if not result['complete'] or len(result['episodes']) != 400:
        raise ValueError('Formal evaluation incomplete')
    run('report_s1.py',['--s1',destination/'formal_run','--t1',args.t1,'--s5',args.s5,
        '--artifacts',args.report_artifacts,'--output',args.output/'report'])
    atomic_json(args.output/'complete.json', dict(selection=best, successes=result['successes'], episodes=400))


if __name__ == '__main__':
    main()
