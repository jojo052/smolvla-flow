#!/usr/bin/env python3
"""Serial train/validate/select orchestration for one approved stage."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets-dir',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--stage',choices=['10to5','5to2'],required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--parent-selection',type=Path)
    p.add_argument('--shared-prefix',action='store_true')
    p.add_argument('--batch16',action='store_true')
    p.add_argument('--cpu-prefetch',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def run(script,options):
        subprocess.run([sys.executable,str(root/'scripts'/script)]+[str(x) for x in options],check=True,cwd=root)
    common=['--checkpoint',args.checkpoint,'--data',args.data,'--output',args.output,
            '--stage',args.stage,'--preflight',args.preflight]
    if args.parent: common+=['--parent',args.parent,'--parent-selection',args.parent_selection]
    if args.shared_prefix: common+=['--shared-prefix']
    if args.batch16: common+=['--batch16']
    if args.cpu_prefetch: common+=['--cpu-prefetch']
    for boundary in ((2500,5000,7500,10000) if args.batch16 else (5000,10000,15000,20000)):
        checkpoint=args.output/f'{boundary:06d}.pt'
        offline=args.output/f'offline_{boundary:06d}.json'
        if not checkpoint.exists() or not offline.exists():
            # A saved checkpoint without offline validation cannot silently skip
            # that validation. Stop for an explicit recovery path in that case.
            if checkpoint.exists(): raise RuntimeError('Checkpoint exists but offline validation is missing')
            saved=sorted(args.output.glob('[0-9][0-9][0-9][0-9][0-9][0-9].pt'))
            options=list(common)
            if saved: options+=['--resume',saved[-1]]
            run('train_distill40_stage.py',options)
        if not checkpoint.exists() or not offline.exists(): raise RuntimeError('Stage did not reach expected boundary')
        validation=args.output/f'closed_loop_{boundary:06d}.json'
        if not validation.exists():
            destination=args.output/f'validation_{boundary:06d}'
            run('evaluate_distill40.py',['--checkpoint',args.checkpoint,'--student',checkpoint,
                '--model','S5' if args.stage=='10to5' else 'S2','--assets-dir',args.assets_dir,
                '--output',destination,'--phase','validation'])
            result=json.loads((destination/'result.json').read_text())
            temporary=validation.with_suffix('.json.partial')
            temporary.write_text(json.dumps(result));temporary.replace(validation)
    run('select_distill40_student.py',['--stage-output',args.output]+(['--batch16'] if args.batch16 else []))

if __name__=='__main__': main()
