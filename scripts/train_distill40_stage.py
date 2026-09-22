#!/usr/bin/env python3
"""One stage of the fixed pure-teacher multi-task experiment.

Stops at each closed-loop selection boundary. The pipeline must validate that
candidate before resuming; formal evaluation never feeds this training entry.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from scripts.distill_smolvla_action_expert import (
    _load_policy, _freeze_teacher, _configure_action_only_student, CachedSmolVLAVelocity,
)
from smolvla_flow.distillation import DistillationConfig
from smolvla_flow.distill40_sampling import ObservationSampler
from smolvla_flow.distill40_observations import ObservationReader, processed_observation, collate_processed
from smolvla_flow.distill40_training import teacher_only_loss, optimizer_update, save_recovery, restore_recovery
from smolvla_flow.distill40_training import shared_prefix_velocity, validate_shared_prefix_models
from smolvla_flow.distill40_selection import validate_selection_episodes
from smolvla_flow.distill40_prefetch import ObservationPrefetch


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def overlay(policy, path, expected_names):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if set(state['trainable'])!=set(expected_names):
        raise ValueError('Parent trainable keys mismatch')
    with torch.no_grad():
        for name,p in policy.named_parameters():
            if name in state['trainable']:
                value=state['trainable'][name]
                if value.shape!=p.shape or value.dtype!=p.dtype: raise ValueError('Parent shape/dtype mismatch')
                p.copy_(value)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['10to5','5to2'],required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--parent-selection',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--shared-prefix',action='store_true',help='Reuse identical frozen teacher prefix; default preserves separate computation')
    p.add_argument('--batch16',action='store_true',help='Independent S5: batch16, 10000 updates, selection every2500')
    p.add_argument('--cpu-prefetch',action='store_true',help='16 GiB single-reader cache and two CPU-decoded lookahead batches')
    args=p.parse_args()
    if args.cpu_prefetch and not args.batch16:
        raise ValueError('CPU prefetch requires batch16')
    if args.batch16 and (args.stage!='10to5' or not args.shared_prefix):
        raise ValueError('Batch16 protocol requires S5 and shared prefix')
    budget=10000 if args.batch16 else 20000
    boundary=2500 if args.batch16 else 5000
    steps=(10,5) if args.stage=='10to5' else (5,2)
    if (args.parent is not None)!=(args.stage=='5to2'): raise ValueError('S2 requires selected S5 parent')
    if args.stage=='5to2':
        if args.parent_selection is None:
            raise ValueError('S2 requires the frozen S5 selection record')
        selection=json.loads(args.parent_selection.read_text())
        if selection['checkpoint_sha256']!=sha(args.parent) or Path(selection['checkpoint']).resolve()!=args.parent.resolve():
            raise ValueError('S2 parent differs from selected S5')
        if {c['update'] for c in selection['candidates']}!={5000,10000,15000,20000}:
            raise ValueError('S5 selection candidates incomplete')
    elif args.parent_selection is not None:
        raise ValueError('First stage cannot use a student parent')
    manifest_path=args.data/'observation_manifest.json'
    provenance={'base_sha256':sha(args.checkpoint/'model.safetensors'),
                'data_manifest_sha256':sha(manifest_path),'stage':args.stage,
                'parent_sha256':sha(args.parent) if args.parent else None}
    gate=json.loads(args.preflight.read_text())
    if gate.get('provenance')!=provenance or gate.get('passed') is not True:
        raise ValueError('Matching real-model preflight required')
    if args.batch16:
        provenance=dict(provenance,training_protocol='s5_batch16_10000_v1')
    seed=123 if args.stage=='10to5' else 124
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    teacher,tp=_load_policy(args.checkpoint,'HuggingFaceVLA/smolvla_libero',steps[0],'cuda')
    student,sp=_load_policy(args.checkpoint,'HuggingFaceVLA/smolvla_libero',steps[1],'cuda')
    names=_configure_action_only_student(student)
    if args.parent:
        overlay(teacher,args.parent,names);overlay(student,args.parent,names)
    _freeze_teacher(teacher)
    if args.shared_prefix: validate_shared_prefix_models(teacher,student)
    params=[v for v in student.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    manifest=json.loads(manifest_path.read_text())
    sampler=ObservationSampler(manifest,seed);reader=ObservationReader(args.data,manifest,max_cache_bytes=16*2**30 if args.cpu_prefetch else None)
    config=DistillationConfig(teacher_steps=steps[0],student_steps=steps[1],action_dim=7)
    args.output.mkdir(parents=True,exist_ok=True)
    start=restore_recovery(args.resume,student,optimizer,sampler,provenance) if args.resume else 0
    with (args.output/'execution_events.jsonl').open('a') as stream:
        stream.write(json.dumps({'event':'start','time_unix':time.time(),
            'resume_update':start,'resume_sha256':sha(args.resume) if args.resume else None,
            'shared_prefix':args.shared_prefix,'microbatch':16 if args.batch16 else 1,'accumulation':1 if args.batch16 else 8,
            'cpu_prefetch':args.cpu_prefetch,
            'training_script_sha256':sha(Path(__file__)),
            'provenance':provenance})+'\n')
    if start and start%boundary==0:
        validation=args.output/f'closed_loop_{start:06d}.json'
        record=json.loads(validation.read_text())
        if record.get('checkpoint_sha256')!=sha(args.resume) or record.get('complete') is not True:
            raise ValueError('Closed-loop validation required before continuing')
        validate_selection_episodes(record['episodes'],sha(args.resume))
    context=torch.zeros(1,1,1,device='cuda')
    def loss_for(row,noise):
        tv=CachedSmolVLAVelocity(teacher,processed_observation(row,tp))
        if args.shared_prefix:
            sv=shared_prefix_velocity(student,tv)
        else:
            sv=CachedSmolVLAVelocity(student,processed_observation(row,sp))
        return teacher_only_loss(sv,tv,noise,context,config)
    aggregates=[]
    prefetch=ObservationPrefetch(sampler,reader) if args.cpu_prefetch else None
    for update in range(start+1,budget+1):
        before=time.monotonic()
        def microbatches():
            if args.batch16:
                if prefetch is not None:
                    rows,_,_=prefetch.next()
                    batch=collate_processed([processed_observation(row,tp,decoded=True) for row in rows])
                else:
                    batch=collate_processed([processed_observation(reader.read(*sampler.sample()),tp) for _ in range(16)])
                noise=torch.cat([torch.randn(1,50,32,device='cuda') for _ in range(16)])
                tv=CachedSmolVLAVelocity(teacher,batch)
                yield teacher_only_loss(shared_prefix_velocity(student,tv),tv,noise,torch.zeros(16,1,1,device='cuda'),config)
                return
            for _ in range(8):
                row=reader.read(*sampler.sample())
                yield loss_for(row,torch.randn(1,50,32,device='cuda'))
        metrics=optimizer_update(params,optimizer,microbatches(),microbatch_size=16 if args.batch16 else 1,effective_batch=16 if args.batch16 else 8)
        metrics.update(update=update,seconds=time.monotonic()-before)
        aggregates.append(metrics)
        if update%100==0:
            record={k:sum(m[k] for m in aggregates)/len(aggregates) for k in metrics if k!='update'}
            record.update(update=update,lr=1e-5,task_samples=dict(sampler.counts),peak_memory=torch.cuda.max_memory_allocated())
            if prefetch is not None:
                import resource
                record.update(reader.stats(),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            with (args.output/'training.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);aggregates=[]
        if update%1000==0 or update%boundary==0:
            if prefetch is not None:
                prefetch.close()  # Join before validation accesses the same reader.
                prefetch=None
            path=args.output/f'{update:06d}.pt'
            save_recovery(path,student,optimizer,sampler,update,provenance)
            student.eval();values=[]
            with torch.no_grad():
                for item in manifest['offline_validation']:
                    row=reader.read(item['task_index'],item['episode_index'],item['frame_index'])
                    for noise_seed in item['noise_seeds']:
                        generator=torch.Generator(device='cuda').manual_seed(noise_seed)
                        out=loss_for(row,torch.randn(1,50,32,device='cuda',generator=generator))
                        values.append({'task':item['task_index'],'loss':float(out.loss),'trajectory':float(out.trajectory_loss),'endpoint':float(out.action_loss)})
            if len(values)!=1280: raise ValueError('Offline validation incomplete')
            (args.output/f'offline_{update:06d}.json').write_text(json.dumps({'checkpoint_sha256':sha(path),'pairs':values,'mean_loss':sum(v['loss'] for v in values)/1280}))
            student.train()
            if update%boundary==0:
                print('CLOSED_LOOP_VALIDATION_REQUIRED',str(path),flush=True)
                return
            if args.cpu_prefetch:
                prefetch=ObservationPrefetch(sampler,reader)

if __name__=='__main__': main()
