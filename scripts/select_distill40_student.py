#!/usr/bin/env python3
"""Select a stage candidate exclusively from held-out closed-loop validation."""
import argparse
import hashlib
import json
from pathlib import Path
from smolvla_flow.distill40_selection import CANDIDATES, select_candidate


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage-output',type=Path,required=True)
    parser.add_argument('--batch16',action='store_true')
    args=parser.parse_args()
    candidates=[]
    schedule=(2500,5000,7500,10000) if args.batch16 else CANDIDATES
    for update in schedule:
        path=args.stage_output/f'{update:06d}.pt'
        sha=file_sha(path)
        closed=json.loads((args.stage_output/f'closed_loop_{update:06d}.json').read_text())
        offline=json.loads((args.stage_output/f'offline_{update:06d}.json').read_text())
        if closed.get('complete') is not True or closed['checkpoint_sha256']!=sha or offline['checkpoint_sha256']!=sha:
            raise ValueError('Candidate validation does not match checkpoint')
        pairs=offline['pairs']
        if len(pairs)!=1280 or any(sum(v['task']==t for v in pairs)!=32 for t in range(40)):
            raise ValueError('Offline validation task coverage mismatch')
        loss=sum(v['loss'] for v in pairs)/len(pairs)
        candidates.append(dict(update=update,checkpoint=str(path.resolve()),checkpoint_sha256=sha,
                               offline_loss=loss,episodes=closed['episodes']))
    selected=select_candidate(candidates,expected_updates=schedule)
    result={k:v for k,v in selected.items() if k!='episodes'}
    result['validation_successes']=sum(v['success'] for v in selected['episodes'])
    result['selection_rule']='successes descending, offline loss ascending, update ascending'
    result['candidates']=[{k:v for k,v in c.items() if k!='episodes'} | {'successes':sum(r['success'] for r in c['episodes'])} for c in candidates]
    target=args.stage_output/'selected.json'
    if target.exists() and json.loads(target.read_text())!=result:
        raise ValueError('Refusing to replace a different frozen selection')
    temporary=target.with_suffix('.json.partial')
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
