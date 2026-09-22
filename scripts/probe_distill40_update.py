#!/usr/bin/env python3
"""Isolated gradient/freeze probe, not a formal student checkpoint."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import pyarrow.parquet as pq
from scripts.distill_smolvla_action_expert import _load_policy, _freeze_teacher, _configure_action_only_student, CachedSmolVLAVelocity
from smolvla_flow.distill40_observations import processed_observation
from smolvla_flow.distill40_training import teacher_only_loss, optimizer_update, save_recovery, restore_recovery
from smolvla_flow.distill40_training import shared_prefix_velocity, validate_shared_prefix_models
from smolvla_flow.distillation import DistillationConfig

def fingerprint(policy, frozen_only=False):
    h=hashlib.sha256()
    for name,p in policy.named_parameters():
        if not frozen_only or not p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--parquet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher-steps',type=int,choices=[10,5],default=10)
    p.add_argument('--parent',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--shared-prefix',action='store_true')
    args=p.parse_args()
    torch.manual_seed(123)
    row=pq.read_table(args.parquet,columns=['observation.images.image','observation.images.image2','observation.state','task_index']).slice(0,1).to_pylist()[0]
    if row['task_index']!=0: raise ValueError('Unexpected task')
    row['task']='put the white mug on the left plate and put the yellow and white mug on the right plate'
    student_steps=5 if args.teacher_steps==10 else 2
    if (args.teacher_steps==5)!=(args.parent is not None):
        raise ValueError('Second-stage probe requires the selected S5 parent')
    teacher,tp=_load_policy(args.checkpoint,'HuggingFaceVLA/smolvla_libero',args.teacher_steps,'cuda')
    student,sp=_load_policy(args.checkpoint,'HuggingFaceVLA/smolvla_libero',student_steps,'cuda')
    _freeze_teacher(teacher);names=_configure_action_only_student(student)
    if args.parent:
        from scripts.train_distill40_stage import overlay
        overlay(teacher,args.parent,names);overlay(student,args.parent,names)
    original_teacher=fingerprint(teacher);original_frozen=fingerprint(student,True)
    original_student=fingerprint(student)
    processed=processed_observation(row,sp)
    perturbed=processed_observation(dict(row,action=[[float('nan')]*7]*50),sp)
    for key,value in processed.items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(value,perturbed[key],rtol=0,atol=0)
    params=[p for p in student.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    context=torch.zeros(1,1,1,device='cuda')
    config=DistillationConfig(teacher_steps=args.teacher_steps,student_steps=student_steps,action_dim=7)
    sampler=None
    if args.data:
        from smolvla_flow.distill40_sampling import ObservationSampler
        from smolvla_flow.distill40_observations import ObservationReader
        manifest=json.loads((args.data/'observation_manifest.json').read_text())
        sampler=ObservationSampler(manifest,123)
        reader=ObservationReader(args.data,manifest)
    if args.resume:
        if sampler is None: raise ValueError('Recovery probe requires data')
        recovery_meta=torch.load(args.resume,map_location='cpu',weights_only=False)['provenance']
        restore_recovery(args.resume,student,optimizer,sampler,recovery_meta)
        original_student=fingerprint(student)
    if args.shared_prefix: validate_shared_prefix_models(teacher,student)
    def losses():
        for _ in range(8):
            observation=reader.read(*sampler.sample()) if sampler else row
            tv=CachedSmolVLAVelocity(teacher,processed_observation(observation,tp))
            if args.shared_prefix:
                a=processed_observation(observation,tp);b=processed_observation(observation,sp)
                for key,value in a.items():
                    if isinstance(value,torch.Tensor): torch.testing.assert_close(value,b[key],rtol=0,atol=0)
                sv=shared_prefix_velocity(student,tv)
            else:
                sv=CachedSmolVLAVelocity(student,processed_observation(observation,sp))
            yield teacher_only_loss(sv,tv,torch.randn(1,50,32,device='cuda'),context,config)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    metrics=optimizer_update(params,optimizer,losses())
    torch.cuda.synchronize();metrics['update_seconds']=time.monotonic()-start
    metrics['peak_memory_bytes']=torch.cuda.max_memory_allocated()
    assert original_teacher==fingerprint(teacher)
    assert original_frozen==fingerprint(student,True)
    assert original_student!=fingerprint(student)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in student.parameters() if not p.requires_grad)
    class ProbeSampler:
        def state_dict(self): return {'probe':True}
        def load_state_dict(self,state):
            if state!={'probe':True}: raise ValueError('Probe state mismatch')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    recovery=args.output.parent/'isolated_recovery.pt'
    recovery_sampler=sampler if sampler else ProbeSampler()
    save_recovery(recovery,student,optimizer,recovery_sampler,1,{'isolated_probe':True})
    uninterrupted=optimizer_update(params,optimizer,losses())
    expected={n:p.detach().cpu().clone() for n,p in student.named_parameters() if p.requires_grad}
    restore_recovery(recovery,student,optimizer,recovery_sampler,{'isolated_probe':True})
    resumed=optimizer_update(params,optimizer,losses())
    maximum=0.0
    for n,p in student.named_parameters():
        if p.requires_grad:
            value=p.detach().cpu()
            maximum=max(maximum,float((value-expected[n]).abs().max()))
            torch.testing.assert_close(value,expected[n],rtol=1e-5,atol=1e-7)
    metrics.update(recovery_max_abs_error=maximum,recovery_probe_passed=True,
                   action_field_ignored=True,continuation_loss=uninterrupted['loss'],resumed_loss=resumed['loss'])
    metrics.update(freeze_probe_passed=True,full_preflight_passed=False,trainable_parameters=sum(p.numel() for p in params),trainable_names=names)
    metrics.update(teacher_steps=args.teacher_steps,student_steps=student_steps,
                   parent_path=str(args.parent) if args.parent else None)
    if args.data:
        from scripts.train_distill40_stage import sha
        metrics['provenance']={'base_sha256':sha(args.checkpoint/'model.safetensors'),
            'data_manifest_sha256':sha(args.data/'observation_manifest.json'),
            'stage':'10to5' if args.teacher_steps==10 else '5to2',
            'parent_sha256':sha(args.parent) if args.parent else None}
        metrics['sample_counts']=dict(sampler.counts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metrics,indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k!='trainable_names'}),flush=True)

if __name__=='__main__': main()
