#!/usr/bin/env python3
"""Fail-closed serial ACT dispatch after a complete, immutable SmolVLA run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl

from smolvla_flow.benchmark40 import sha256, fingerprint, validate_episode, summarize

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/root/autodl-tmp')
PREVIOUS = BASE / 'outputs/libero40_public_v1'
OUTPUT = BASE / 'outputs/libero40_act_v1'
CHECKPOINT = BASE / 'checkpoints/public_act/jamongsteak-act_libero-8744e679'


def ready(status):
    if status.get('status') == 'error':
        raise RuntimeError('SmolVLA reported an execution error; ACT will not start')
    return status.get('status') == 'complete' and status.get('formal_complete') == 400


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('WAITING for verified SmolVLA completion and evaluator lock release', flush=True)
        while True:
            status = json.loads((PREVIOUS / 'status.json').read_text())
            if ready(status):
                with (PREVIOUS / '.lock').open('r') as previous_lock:
                    try:
                        fcntl.flock(previous_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(30)
                        continue
                    manifest = json.loads((PREVIOUS / 'manifest.json').read_text())
                    # Recompute validated unique episode counts and state hashes.
                    rows = []
                    for path in sorted((PREVIOUS / 'formal').glob('*/*/task*/init*.json')):
                        row = json.loads(path.read_text())
                        validate_episode(row, fingerprint(manifest), str(path.relative_to(PREVIOUS).with_suffix('')))
                        task = next(t for t in manifest['tasks'] if t['suite'] == row['suite'] and t['task_id'] == row['task_id'])
                        if row['init_state_sha256'] != task['state_sha256'][row['init_index']]:
                            raise RuntimeError('Initial state mismatch')
                        rows.append(row)
                    summary = summarize(rows, ['smolvla_official'])
                    if len(rows) != 400 or not summary['models']['smolvla_official']['complete']:
                        raise RuntimeError('Predecessor does not contain exactly 400 valid formal episodes')
                    checkpoint = Path(manifest['checkpoint'])
                    if {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()} != manifest['checkpoint_files']:
                        raise RuntimeError('SmolVLA checkpoint changed')
                    break
            time.sleep(30)
        audit = json.loads((BASE / 'outputs/libero40_act_audit_v1/audit.json').read_text())
        if audit['status'] != 'audit_passed_with_scope_caveats':
            raise RuntimeError('ACT audit not passed')
        for name, info in audit['files'].items():
            if sha256(CHECKPOINT / name) != info['sha256']:
                raise RuntimeError('ACT audited file changed: ' + name)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('GPU compute processes remain; refusing concurrent evaluation: ' + active)
        print('START ACT: four pilot episodes, then 400 formal episodes', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_libero40_act.py'),
            '--checkpoint', str(CHECKPOINT), '--assets-dir', str(BASE / 'libero-assets'),
            '--output', str(OUTPUT), '--phase', 'all'], check=True)
        print('ACT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
