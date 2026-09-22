"""Pause one identified trainer only after its 1000-update artifacts validate."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import math


def identity(pid):
    proc = Path('/proc') / str(pid)
    return ((proc / 'stat').read_text().split(') ', 1)[1].split()[19],
            (proc / 'cmdline').read_bytes())


def main():
    pid = 14018
    root = Path('/root/autodl-tmp/outputs/distill40_v1/stage1')
    original = identity(pid)
    if b'train_distill40_stage.py' not in original[1] or b'10to5' not in original[1]:
        raise RuntimeError('Unexpected trainer identity')
    print('WATCHING', pid, original[0], flush=True)
    checkpoint = root / '001000.pt'
    validation = root / 'offline_001000.json'
    while True:
        if identity(pid) != original:
            raise RuntimeError('Trainer exited or identity changed; no signal sent')
        if checkpoint.is_file() and validation.is_file():
            try:
                result = json.loads(validation.read_text())
            except json.JSONDecodeError:
                time.sleep(5)
                continue
            pairs = result['pairs']
            counts = {}
            if len(pairs) != 1280:
                raise RuntimeError('Incomplete validation')
            for row in pairs:
                counts[row['task']] = counts.get(row['task'], 0) + 1
                if not all(math.isfinite(row[k]) for k in ('loss', 'trajectory', 'endpoint')):
                    raise RuntimeError('Nonfinite validation')
            if len(counts) != 40 or set(counts.values()) != {32}:
                raise RuntimeError('Invalid task coverage')
            digest = hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != result['checkpoint_sha256']:
                raise RuntimeError('Checkpoint hash mismatch')
            if identity(pid) != original:
                raise RuntimeError('Trainer changed before pause')
            os.kill(pid, signal.SIGSTOP)
            time.sleep(1)
            state = (Path('/proc') / str(pid) / 'status').read_text()
            if '\nState:\tT' not in state:
                raise RuntimeError('Pause not confirmed')
            record = {'pid': pid, 'start_ticks': original[0], 'checkpoint_sha256': digest.hexdigest(),
                      'validation_pairs': len(pairs), 'paused_at_unix': time.time(), 'signal': 'SIGSTOP'}
            (root / 'pause_001000.json').write_text(json.dumps(record, indent=2))
            print('PAUSED', json.dumps(record), flush=True)
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
