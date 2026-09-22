"""One-shot handoff from the running RTC check to formal evaluation.

Waits on the check's advisory lock, not assistant polling. No retries or power control.
"""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','checkpoint','student','assets-dir','baseline-manifest','protocol'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if not (args.output/'run.lock').is_file():raise RuntimeError('No check has been started')
    with (args.output/'handoff.lock').open('a') as once:
        fcntl.flock(once,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (args.output/'run.lock').open('a') as lock:
            print('Waiting for check process to release its lock',flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
            if (args.output/'error.json').exists():raise RuntimeError('Check failed; formal evaluation will not start')
            gate=json.loads((args.output/'gate.json').read_text())
            if not gate.get('passed') or gate.get('episodes')!=24:raise RuntimeError('Missing 24-episode gate')
        command=[sys.executable,'-u','-m','scripts.evaluate_s1_rtc','--stage','run']
        for name,value in vars(args).items():command += ['--'+name.replace('_','-'),str(value)]
        print('Checks passed; starting formal evaluation',flush=True)
        subprocess.run(command,check=True)


if __name__=='__main__':main()
