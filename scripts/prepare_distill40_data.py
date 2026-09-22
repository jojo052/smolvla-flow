#!/usr/bin/env python3
"""Materialize pinned data and derive episode membership from actual rows.

Source metadata is retained unchanged. Derived indices never trust its broken
file offsets. Expert actions are not read while building the observation index.
"""
import argparse
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

REVISION = '86958911c0f959db2bbbdb107eb3e17c5f9c798e'
HOST = 'https://hf-mirror.com'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--benchmark-manifest', type=Path, required=True)
    p.add_argument('--footer-audit', type=Path, required=True)
    args = p.parse_args()
    import pyarrow.parquet as pq
    tree = json.loads(subprocess.check_output(['curl','-fLsS','--max-time','60',f'{HOST}/api/datasets/HuggingFaceVLA/libero/tree/{REVISION}?recursive=true&limit=1000']))
    files = [x for x in tree if x['type']=='file' and (x['path'].startswith('data/') or x['path'].startswith('meta/'))]
    data_files = sorted([x for x in files if x['path'].startswith('data/')],key=lambda x:x['path'])
    if len(data_files)!=377:
        raise ValueError('Unexpected pinned file listing')
    end = 0
    for item in data_files:
        a = json.loads((args.footer_audit/(item['path'].replace('/','__')+'.json')).read_text())
        start = min(g['index'][0] for g in a['groups'])
        stop = max(g['index'][1] for g in a['groups'])+1
        if start!=end or stop-start!=a['rows'] or a['size']!=item['size']:
            raise ValueError(f'Non-contiguous footer indices: {item["path"]}')
        end=stop
    if end!=273465:
        raise ValueError('Incomplete footer frame coverage')
    args.root.mkdir(parents=True,exist_ok=True)
    hashes = {}
    for item in sorted(files,key=lambda x:x['path']):
        dest=args.root/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        expected=item.get('lfs',{}).get('oid')
        if not dest.exists():
            import shutil
            if shutil.disk_usage(args.root).free < item['size']+5*1024**3:
                raise RuntimeError('Insufficient disk headroom')
            temp=dest.with_suffix(dest.suffix+'.partial')
            subprocess.run(['curl','-fL','--retry','3','--connect-timeout','20','--max-time','1800','-C','-','-o',str(temp),f'{HOST}/datasets/HuggingFaceVLA/libero/resolve/{REVISION}/{item["path"]}'],check=True)
            if temp.stat().st_size!=item['size']:
                raise ValueError('Downloaded size mismatch')
            if expected and digest(temp)!=expected:
                raise ValueError('Downloaded LFS SHA256 mismatch')
            temp.rename(dest)
        actual=digest(dest)
        if dest.stat().st_size!=item['size'] or (expected and actual!=expected):
            raise ValueError(f'Existing file mismatch: {dest}')
        hashes[item['path']]=actual
        print('verified',item['path'],flush=True)
    tasks=pq.read_table(args.root/'meta/tasks.parquet').to_pylist()
    languages={int(t['task_index']):t['__index_level_0__'] for t in tasks}
    benchmark=json.loads(args.benchmark_manifest.read_text())
    by_language={t['language']:t for t in benchmark['tasks']}
    if len(languages)!=40 or set(languages.values())!=set(by_language):
        raise ValueError('Task identity mismatch')
    episodes={};next_index=0
    for item in data_files:
        rows=pq.read_table(args.root/item['path'],columns=['index','episode_index','task_index','frame_index']).to_pylist()
        for offset,row in enumerate(rows):
            if row['index']!=next_index:
                raise ValueError('Actual row index gap or duplicate')
            next_index+=1
            ep=int(row['episode_index']);task=int(row['task_index'])
            e=episodes.setdefault(ep,{'episode_index':ep,'task_index':task,'frames':[]})
            if e['task_index']!=task or row['frame_index']!=len(e['frames']):
                raise ValueError('Episode task or frame order mismatch')
            e['frames'].append([item['path'],offset])
    if len(episodes)!=1693 or next_index!=273465:
        raise ValueError('Episode/frame count mismatch')
    source=pq.read_table(args.root/'meta/episodes/chunk-000/file-000.parquet',columns=['episode_index','length','tasks']).to_pylist()
    for row in source:
        e=episodes[int(row['episode_index'])]
        if len(e['frames'])!=row['length'] or row['tasks']!=[languages[e['task_index']]]:
            raise ValueError('Episode length or instruction mismatch')
    groups=defaultdict(list)
    for ep,e in episodes.items(): groups[e['task_index']].append(ep)
    splits={};validation=[]
    for task in sorted(groups):
        ids=sorted(groups[task]);rng=random.Random(123);rng.shuffle(ids)
        n=max(1,round(len(ids)*0.1));val=ids[:n];train=ids[n:]
        if not train: raise ValueError('Empty training partition')
        splits[str(task)]={'train':train,'validation':val,'language':languages[task], 'suite':by_language[languages[task]]['suite'],'task_id':by_language[languages[task]]['task_id']}
        candidates=[(ep,i) for ep in val for i in range(len(episodes[ep]['frames']))]
        for ep,i in random.Random(123+task).sample(candidates,16):
            validation.append({'task_index':task,'episode_index':ep,'frame_index':i,'noise_seeds':[123000+len(validation)*2,123001+len(validation)*2]})
    result={'revision':REVISION,'file_sha256':hashes,'episodes':list(episodes.values()),'splits':splits,'offline_validation':validation,'source_offsets_used':False,'expert_actions_used':False}
    temp=args.root/'observation_manifest.json.partial';temp.write_text(json.dumps(result))
    temp.rename(args.root/'observation_manifest.json')
    print('DATA_READY',len(episodes),next_index,flush=True)

if __name__=='__main__': main()
