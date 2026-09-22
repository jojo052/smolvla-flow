#!/usr/bin/env python3
"""Wait for one identified stage-one pipeline, then evaluate its frozen S5."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return None
    return stat.rsplit(')', 1)[1].split()[19], command


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pipeline-pid', type=int, required=True)
    p.add_argument('--stage-output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--assets-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch16',action='store_true')
    args = p.parse_args()
    original = identity(args.pipeline_pid)
    if original is None or 'run_distill40_stage_pipeline.py' not in original[1] or '--stage 10to5' not in original[1]:
        raise ValueError('Expected a live stage-one pipeline')
    print('WAITING_FOR_PIPELINE', args.pipeline_pid, original[0], flush=True)
    while identity(args.pipeline_pid) == original:
        time.sleep(60)
    selected = json.loads((args.stage_output/'selected.json').read_text())
    schedule={2500,5000,7500,10000} if args.batch16 else {5000,10000,15000,20000}
    if {item['update'] for item in selected['candidates']} != schedule:
        raise ValueError('Incomplete selection candidates')
    student = Path(selected['checkpoint'])
    if student.resolve().parent != args.stage_output.resolve():
        raise ValueError('Selected checkpoint outside stage output')
    digest = hashlib.sha256()
    with student.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash changed')
    root = Path(__file__).resolve().parents[1]
    print('STARTING_FORMAL_S5', str(student), flush=True)
    subprocess.run([sys.executable, str(root/'scripts/evaluate_distill40.py'),
        '--checkpoint', str(args.checkpoint), '--student', str(student), '--model', 'S5',
        '--assets-dir', str(args.assets_dir), '--output', str(args.output), '--phase', 'formal'],
        check=True, cwd=root)
    print('FORMAL_S5_FINISHED', flush=True)


if __name__ == '__main__':
    main()
