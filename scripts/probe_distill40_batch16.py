"""Isolated batch-16 timing: one warmup and five measured optimizer updates."""
import argparse
import json
import time
from pathlib import Path
import torch
from scripts.probe_distill40_microbatch import collate
from scripts.probe_distill40_update import fingerprint
from scripts.distill_smolvla_action_expert import _load_policy, _freeze_teacher, _configure_action_only_student, CachedSmolVLAVelocity
from smolvla_flow.distill40_training import restore_recovery, teacher_only_loss, shared_prefix_velocity, validate_shared_prefix_models
from smolvla_flow.distill40_observations import ObservationReader, processed_observation
from smolvla_flow.distill40_sampling import ObservationSampler
from smolvla_flow.distillation import DistillationConfig


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','resume','data','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    teacher,tp=_load_policy(a.checkpoint,'HuggingFaceVLA/smolvla_libero',10,'cuda')
    student,_=_load_policy(a.checkpoint,'HuggingFaceVLA/smolvla_libero',5,'cuda')
    _freeze_teacher(teacher);_configure_action_only_student(student)
    params=[v for v in student.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    manifest=json.loads((a.data/'observation_manifest.json').read_text())
    sampler=ObservationSampler(manifest,123);reader=ObservationReader(a.data,manifest)
    state=torch.load(a.resume,map_location='cpu',weights_only=False)
    provenance=state['provenance'];del state
    restore_recovery(a.resume,student,opt,sampler,provenance)
    validate_shared_prefix_models(teacher,student)
    original=fingerprint(student,True); teacher_original=fingerprint(teacher)
    config=DistillationConfig(teacher_steps=10,student_steps=5,action_dim=7)
    report={'microbatch':16,'accumulation':1,'effective_batch':16,'updates':[],'source':str(a.resume)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for step in range(6):
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        batch=collate([processed_observation(reader.read(*sampler.sample()),tp) for _ in range(16)])
        noise=torch.cat([torch.randn(1,50,32,device='cuda') for _ in range(16)])
        tv=CachedSmolVLAVelocity(teacher,batch);sv=shared_prefix_velocity(student,tv)
        opt.zero_grad(set_to_none=True)
        out=teacher_only_loss(sv,tv,noise,torch.zeros(16,1,1,device='cuda'),config)
        out.loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        opt.step()
        if any(not torch.isfinite(v).all() for v in params): raise FloatingPointError('Nonfinite parameter')
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        row={'warmup':step==0,'seconds':elapsed,'samples_per_second':16/elapsed,'loss':float(out.loss),
             'gradient_norm':float(norm),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
        report['updates'].append(row)
        a.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        del out,tv,sv,batch,noise
    report['frozen_unchanged']=original==fingerprint(student,True) and teacher_original==fingerprint(teacher)
    report['mean_seconds']=sum(x['seconds'] for x in report['updates'][1:])/5
    report['complete']=True
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
