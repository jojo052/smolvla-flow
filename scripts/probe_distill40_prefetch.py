"""Isolated identical-sequence optimizer benchmark and recovery gate."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import torch
from scripts.distill_smolvla_action_expert import _load_policy, _freeze_teacher, _configure_action_only_student, CachedSmolVLAVelocity
from scripts.probe_distill40_update import fingerprint
from smolvla_flow.distill40_observations import ObservationReader, processed_observation, collate_processed
from smolvla_flow.distill40_prefetch import ObservationPrefetch
from smolvla_flow.distill40_sampling import ObservationSampler
from smolvla_flow.distill40_training import restore_recovery, save_recovery, teacher_only_loss, optimizer_update, shared_prefix_velocity, validate_shared_prefix_models
from smolvla_flow.distillation import DistillationConfig


def digest(batch, noise):
    h=hashlib.sha256()
    for key,value in sorted(dict(batch,noise=noise).items()):
        if isinstance(value,torch.Tensor):
            h.update(key.encode());h.update(str((value.dtype,value.shape)).encode())
            h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','resume','data','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    teacher,tp=_load_policy(a.checkpoint,'HuggingFaceVLA/smolvla_libero',10,'cuda')
    student,_=_load_policy(a.checkpoint,'HuggingFaceVLA/smolvla_libero',5,'cuda')
    _freeze_teacher(teacher);_configure_action_only_student(student)
    validate_shared_prefix_models(teacher,student)
    params=[p for p in student.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    manifest=json.loads((a.data/'observation_manifest.json').read_text())
    state=torch.load(a.resume,map_location='cpu',weights_only=False)
    provenance=state['provenance'];del state
    sampler=ObservationSampler(manifest,123)
    config=DistillationConfig(teacher_steps=10,student_steps=5,action_dim=7)
    frozen=fingerprint(student,True);teacher_hash=fingerprint(teacher)
    report={'source':str(a.resume),'tolerance':{'rtol':1e-5,'atol':1e-7},'runs':{}}
    expected=None
    for mode in ('baseline','prefetch'):
        restore_recovery(a.resume,student,opt,sampler,provenance)
        reader=ObservationReader(a.data,manifest,max_cache_bytes=16*2**30 if mode=='prefetch' else None)
        pre=ObservationPrefetch(sampler,reader) if mode=='prefetch' else None
        rows=[]
        def update(check_input=False):
            start=time.monotonic()
            if pre is None:
                keys=[sampler.sample() for _ in range(16)]
                batch=collate_processed([processed_observation(reader.read(*k),tp) for k in keys])
            else:
                obs,keys,_=pre.next()
                batch=collate_processed([processed_observation(x,tp,decoded=True) for x in obs])
            prepared=time.monotonic()-start
            noise=torch.cat([torch.randn(1,50,32,device='cuda') for _ in range(16)])
            # Hash outside timed region, so CPU transfer is not reported as training time.
            hashed=digest(batch,noise) if check_input else None
            torch.cuda.synchronize();compute=time.monotonic()
            tv=CachedSmolVLAVelocity(teacher,batch)
            out=teacher_only_loss(shared_prefix_velocity(student,tv),tv,noise,torch.zeros(16,1,1,device='cuda'),config)
            metrics=optimizer_update(params,opt,[out],microbatch_size=16,effective_batch=16)
            torch.cuda.synchronize()
            metrics.update(seconds=prepared+time.monotonic()-compute,prepare_seconds=prepared,input_hash=hashed,keys=keys)
            return metrics
        try:
            for i in range(20):
                row=update(check_input=i==0);rows.append(row)
                print(json.dumps({'mode':mode,'update':i+1,'seconds':row['seconds'],'loss':row['loss']}),flush=True)
            actual={n:p.detach().cpu().clone() for n,p in student.named_parameters() if p.requires_grad}
            if mode=='baseline': expected=actual
            else:
                assert [x['input_hash'] for x in rows]==[x['input_hash'] for x in report['runs']['baseline']['updates']]
                assert [x['keys'] for x in rows]==[x['keys'] for x in report['runs']['baseline']['updates']]
                for n,v in actual.items(): torch.testing.assert_close(v,expected[n],rtol=1e-5,atol=1e-7)
                # Save with two lookahead batches pending, then reproduce one update.
                save_recovery(a.output/'isolated.pt',student,opt,sampler,20,provenance)
                uninterrupted=update(check_input=True)
                wanted={n:p.detach().cpu().clone() for n,p in student.named_parameters() if p.requires_grad}
                pre.close()
                restore_recovery(a.output/'isolated.pt',student,opt,sampler,provenance)
                pre=ObservationPrefetch(sampler,reader)
                resumed=update(check_input=True)
                assert uninterrupted['input_hash']==resumed['input_hash'] and uninterrupted['keys']==resumed['keys']
                for n,v in student.named_parameters():
                    if v.requires_grad: torch.testing.assert_close(v.detach().cpu(),wanted[n],rtol=1e-5,atol=1e-7)
                report['recovery_passed']=True
        finally:
            if pre is not None: pre.close()
        report['runs'][mode]={'updates':rows,'mean_seconds':sum(x['seconds'] for x in rows[2:])/18,
                             'cache':reader.stats(),'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        del reader
    report['frozen_unchanged']=frozen==fingerprint(student,True) and teacher_hash==fingerprint(teacher)
    report['speedup']=report['runs']['baseline']['mean_seconds']/report['runs']['prefetch']['mean_seconds']
    report['passed']=report['frozen_unchanged'] and report['recovery_passed'] and report['speedup']>1.05
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print('RESULT',report['passed'],report['speedup'],flush=True)

if __name__=='__main__': main()
