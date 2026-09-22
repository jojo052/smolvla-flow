#!/usr/bin/env python3
"""Recompute four-group results only after all formal episodes are present."""
import argparse
import json
from pathlib import Path
from smolvla_flow.benchmark40 import summarize, markdown_report, fingerprint, atomic_json, validate_episode

def main():
    p=argparse.ArgumentParser()
    for model in ('T10','T2','S5','S2'): p.add_argument('--'+model.lower(),type=Path)
    p.add_argument('--s5-only',action='store_true',help='Report the approved T10/S5 comparison; default remains all four groups')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[];manifests={}
    models=['T10','S5'] if args.s5_only else ['T10','T2','S5','S2']
    for model in models:
        if getattr(args,model.lower()) is None: p.error('--'+model.lower()+' is required')
    if args.s5_only and (args.t2 is not None or args.s2 is not None):
        p.error('--s5-only cannot silently ignore T2 or S2 inputs')
    expected={(suite,t,i) for suite in ('libero_spatial','libero_object','libero_goal','libero_10') for t in range(10) for i in range(10)}
    for model in models:
        folder=getattr(args,model.lower());manifest=json.loads((folder/'manifest.json').read_text())
        manifests[model]=manifest;seen=set()
        for path in sorted((folder/'formal').glob('*/*/task*/init*.json')):
            row=json.loads(path.read_text());identity=(row['suite'],row['task_id'],row['init_index'])
            validate_episode(row,fingerprint(manifest),path.relative_to(folder).with_suffix('').as_posix())
            if identity not in expected or identity in seen: raise ValueError('Unexpected/duplicate formal episode')
            if row['run_id']!=fingerprint(manifest) or row['status']!='complete' or row['phase']!='formal': raise ValueError('Invalid formal provenance')
            task=next(t for t in manifest['tasks'] if (t['suite'],t['task_id'])==identity[:2])
            if row['init_state_sha256']!=task['state_sha256'][identity[2]]: raise ValueError('Initial state mismatch')
            if row['action_nonfinite_count'] or len(row['prediction_seconds'])!=row['steps']: raise ValueError('Every-step finite prediction required')
            seen.add(identity);rows.append(dict(row,model=model))
        if seen!=expected: raise ValueError(f'{model}: incomplete, {len(seen)}/400')
    baseline=manifests['T10']
    for model,m in manifests.items():
        for key in ('suites','settle_steps','policy_seed','environment_seed','mode','rtc','observation_size','action_processing'):
            if baseline['protocol'][key]!=m['protocol'][key]: raise ValueError('Protocol mismatch '+key)
        for task in m['tasks']:
            other=next(t for t in baseline['tasks'] if (t['suite'],t['task_id'])==(task['suite'],task['task_id']))
            for key in ('name','language','bddl_sha256','init_file_sha256','max_steps'):
                if task[key]!=other[key]: raise ValueError('Task identity changed')
            if task['state_sha256'][:10]!=other['state_sha256'][:10]: raise ValueError('Paired states changed')
    result=summarize(rows,models)
    pairs=[('S5','T10')] if args.s5_only else [('S2','T2'),('S5','T10'),('S2','S5'),('S2','T10')]
    result['paired_comparisons']={a+'-'+b:summarize(rows,[a,b])['paired'] for a,b in pairs}
    result['source_manifests']=manifests
    atomic_json(args.output/'results.json',result)
    report=markdown_report(result)+'\n\nT10 reuses an earlier run. These are fixed comparisons on previously viewed test conditions.\n'
    (args.output/'results.md').write_text(report)

if __name__=='__main__': main()
