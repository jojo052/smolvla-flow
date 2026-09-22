#!/usr/bin/env python3
"""Build a training gate only from checked data/model/preflight evidence."""
import argparse
import json
from pathlib import Path
from scripts.train_distill40_stage import sha
from smolvla_flow.distill40_sampling import ObservationSampler

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--benchmark-manifest',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--euler',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    probe=json.loads(args.probe.read_text());euler=json.loads(args.euler.read_text())
    for flag in ('recovery_probe_passed','freeze_probe_passed','action_field_ignored'):
        if probe.get(flag) is not True: raise ValueError('Missing probe: '+flag)
    if probe['provenance']['stage']!='10to5': raise ValueError('This gate currently covers first stage only')
    if euler.get('euler_probe_passed') is not True or {r['steps'] for r in euler['results']}!={10,5,2}:
        raise ValueError('Missing Euler check')
    if any(r['max_abs_error']>1e-6 for r in euler['results']): raise ValueError('Euler mismatch')
    pilot=json.loads(args.pilot.read_text())
    if pilot.get('complete') is not True or len(pilot['episodes'])!=4: raise ValueError('Pilot incomplete')
    if any(r['action_nonfinite_count'] or r['init_index']!=10 for r in pilot['episodes']): raise ValueError('Pilot contract mismatch')
    official=json.loads(args.benchmark_manifest.read_text())
    for name,expected in official['checkpoint_files'].items():
        if sha(args.checkpoint/name)!=expected: raise ValueError('Official file changed: '+name)
    manifest_path=args.data/'observation_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if sha(manifest_path)!=probe['provenance']['data_manifest_sha256']: raise ValueError('Data manifest changed')
    if sha(args.checkpoint/'model.safetensors')!=probe['provenance']['base_sha256']: raise ValueError('Base model changed')
    ObservationSampler(manifest,123)
    if len(manifest['episodes'])!=1693 or sum(len(e['frames']) for e in manifest['episodes'])!=273465: raise ValueError('Data count mismatch')
    if len(manifest['offline_validation'])!=640: raise ValueError('Offline observations incomplete')
    for name,expected in manifest['file_sha256'].items():
        if sha(args.data/name)!=expected: raise ValueError('Data file changed: '+name)
    result={'passed':True,'provenance':probe['provenance'],
            'evidence':{str(path):sha(path) for path in (args.probe,args.euler,args.pilot,args.benchmark_manifest)},
            'scope':'10to5 training; second stage requires its own selected-parent preflight'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print('PREFLIGHT_APPROVED',args.output,flush=True)

if __name__=='__main__': main()
