#!/usr/bin/env python3
"""Isolated real-model Euler probe, never a full training authorization gate."""
import argparse
import json
import time
from pathlib import Path
import torch
import pyarrow.parquet as pq
from scripts.distill_smolvla_action_expert import _load_policy, _freeze_teacher, CachedSmolVLAVelocity
from smolvla_flow.distill40_observations import processed_observation
from smolvla_flow.distillation import euler_rollout


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--parquet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.manual_seed(123)
    row=pq.read_table(args.parquet,columns=['observation.images.image','observation.images.image2','observation.state','task_index']).slice(0,1).to_pylist()[0]
    if row['task_index']!=0: raise ValueError('Probe expects pinned dataset first task')
    row['task']='put the white mug on the left plate and put the yellow and white mug on the right plate'
    policy,processor=_load_policy(args.checkpoint,'HuggingFaceVLA/smolvla_libero',10,'cuda')
    _freeze_teacher(policy)
    batch=processed_observation(row,processor)
    images,masks=policy.prepare_images(batch);state=policy.prepare_state(batch)
    noise=torch.randn(1,50,32,device='cuda')
    results=[]
    with torch.no_grad():
        for steps in (10,5,2):
            policy.config.num_steps=steps;policy.model.config.num_steps=steps
            torch.cuda.synchronize();start=time.monotonic()
            official=policy.model.sample_actions(images,masks,batch['observation.language.tokens'],batch['observation.language.attention_mask'],state,noise=noise.clone())
            adapter=CachedSmolVLAVelocity(policy,batch)
            custom=euler_rollout(adapter,noise.clone(),torch.zeros(1,1,1,device='cuda'),num_steps=steps,track_grad=False)[:,-1]
            repeated=euler_rollout(adapter,noise.clone(),torch.zeros(1,1,1,device='cuda'),num_steps=steps,track_grad=False)[:,-1]
            torch.cuda.synchronize()
            torch.testing.assert_close(custom,official,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(custom,repeated,rtol=0,atol=0)
            if not torch.isfinite(custom).all(): raise ValueError('Non-finite actions')
            results.append({'steps':steps,'max_abs_error':float((custom-official).abs().max()),'shape':list(custom.shape),'three_rollouts_seconds':time.monotonic()-start})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'euler_probe_passed':True,'full_preflight_passed':False,'results':results},indent=2))
    print(args.output.read_text(),flush=True)

if __name__=='__main__': main()
