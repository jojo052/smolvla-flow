"""Isolated T2->S1 probe and fixed-sample training. Never changes old entries."""
import argparse
import copy
import hashlib
import json
import random
import time
import resource
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from scripts.distill_smolvla_action_expert import (_load_policy, _freeze_teacher,
    _configure_action_only_student, CachedSmolVLAVelocity)
from smolvla_flow.distillation import DistillationConfig, euler_rollout
from smolvla_flow.distill40_training import (teacher_only_loss, shared_prefix_velocity,
    validate_shared_prefix_models, save_recovery, restore_recovery)
from smolvla_flow.distill40_observations import (ObservationReader, processed_observation,
    collate_processed, decode_observation)
from smolvla_flow.distill40_sampling import ObservationSampler
from smolvla_flow.distill40_prefetch import ObservationPrefetch
from smolvla_flow.benchmark40 import atomic_json, sha256, OFFICIAL_SHA256
from scripts.s1_resources import ResourceLog


class Prefetch(ObservationPrefetch):
    def __init__(self,sampler,reader,batch_size,decode_workers=1):
        self.decoder=ThreadPoolExecutor(max_workers=2) if decode_workers==2 else None
        super().__init__(sampler,reader,batch_size=batch_size,depth=2)
    def _prepare(self,requests):
        raw=[self.reader.read(*key) for key in requests]  # sole cache owner
        rows=list(self.decoder.map(decode_observation,raw)) if self.decoder else [decode_observation(r) for r in raw]
        return rows,self.reader.stats()
    def close(self):
        super().close()
        if self.decoder: self.decoder.shutdown(wait=True,cancel_futures=True)


class TimedVelocity:
    """CUDA event timings do not change the velocity function or its gradients."""
    def __init__(self, velocity):
        self.velocity=velocity;self.events=[]
    def __call__(self,*args):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();value=self.velocity(*args);end.record()
        self.events.append((start,end))
        return value
    def seconds(self):
        return sum(a.elapsed_time(b) for a,b in self.events)/1000


def fingerprint(model,frozen_only=False):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if frozen_only and p.requires_grad: continue
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Engine:
    def __init__(self,args):
        self.args=args
        random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
        torch.set_num_threads(4)
        self.manifest=json.loads((args.data/'observation_manifest.json').read_text())
        self.provenance=dict(stage='2to1',base_sha256=sha256(args.checkpoint/'model.safetensors'),
            data_manifest_sha256=sha256(args.data/'observation_manifest.json'),batch=args.batch,
            sample_budget=160000,seed=123,teacher_steps=2,student_steps=1,
            decode_workers=args.decode_workers,entry_sha256=sha256(Path(__file__)))
        if self.provenance['base_sha256']!=OFFICIAL_SHA256: raise ValueError('Official identity changed')
        self.teacher,self.processor=_load_policy(args.checkpoint,'HuggingFaceVLA/smolvla_libero',2,'cuda')
        self.student,_=_load_policy(args.checkpoint,'HuggingFaceVLA/smolvla_libero',1,'cuda')
        _freeze_teacher(self.teacher); self.names=_configure_action_only_student(self.student)
        checkpoint_files={str(p.relative_to(args.checkpoint)):sha256(p) for p in sorted(args.checkpoint.rglob('*'))
                          if p.is_file() and p.name!='model.safetensors'}
        checkpoint_files['model.safetensors']=self.provenance['base_sha256']
        atomic_json(args.output/'identity.json',dict(provenance=self.provenance,
            checkpoint_files=checkpoint_files,trainable_names=self.names,
            torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0)))
        validate_shared_prefix_models(self.teacher,self.student)
        self.params=[p for p in self.student.parameters() if p.requires_grad]
        self.optimizer=torch.optim.AdamW(self.params,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
        self.sampler=ObservationSampler(self.manifest,123)
        self.reader=ObservationReader(args.data,self.manifest,max_cache_bytes=16*2**30)
        self.config=DistillationConfig(teacher_steps=2,student_steps=1,action_dim=7)
        self.prefetch=None;self.update=0
    def start_prefetch(self):
        self.prefetch=Prefetch(self.sampler,self.reader,self.args.batch,self.args.decode_workers)
    def close_prefetch(self):
        if self.prefetch: self.prefetch.close();self.prefetch=None
    def checkpoint(self,path):
        self.close_prefetch()
        save_recovery(path,self.student,self.optimizer,self.sampler,self.update,self.provenance)
    def restore(self,path):
        self.close_prefetch()
        self.update=restore_recovery(path,self.student,self.optimizer,self.sampler,self.provenance)
    def batch(self):
        if self.prefetch is None: self.start_prefetch()
        t=time.perf_counter();rows,keys,stats=self.prefetch.next(); wait_s=time.perf_counter()-t
        t=time.perf_counter()
        batch=collate_processed([processed_observation(row,self.processor,decoded=True) for row in rows])
        torch.cuda.synchronize();prep_s=time.perf_counter()-t
        return batch,keys,stats,wait_s,prep_s
    def step(self):
        self.resources.check()
        started=time.perf_counter();batch,keys,stats,wait_s,prep_s=self.batch()
        noise=torch.randn(self.args.batch,50,32,device='cuda')
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();t=time.perf_counter()
        tv=CachedSmolVLAVelocity(self.teacher,batch)
        torch.cuda.synchronize();prefix_s=time.perf_counter()-t;t=time.perf_counter()
        timed_teacher=TimedVelocity(tv);timed_student=TimedVelocity(shared_prefix_velocity(self.student,tv))
        out=teacher_only_loss(timed_student,timed_teacher,noise,
                              torch.zeros(self.args.batch,1,1,device='cuda'),self.config)
        torch.cuda.synchronize();forward_s=time.perf_counter()-t;t=time.perf_counter()
        assert len(timed_teacher.events)==2 and len(timed_student.events)==1
        out.loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.params,1.,error_if_nonfinite=True)
        torch.cuda.synchronize();backward_s=time.perf_counter()-t;t=time.perf_counter()
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.params): raise FloatingPointError('Nonfinite parameter')
        if any(p.grad is not None for p in self.teacher.parameters()): raise ValueError('Teacher gradient')
        torch.cuda.synchronize();opt_s=time.perf_counter()-t
        self.update+=1
        row=dict(update=self.update,samples=self.update*self.args.batch,loss=float(out.loss),
            endpoint=float(out.action_loss),trajectory=float(out.trajectory_loss),gradient_norm=float(norm),
            lr=1e-5,seconds=time.perf_counter()-started,wait_s=wait_s,preprocess_s=prep_s,
            prefix_s=prefix_s,teacher_student_forward_s=forward_s,backward_s=backward_s,optimizer_s=opt_s,
            teacher_velocity_s=timed_teacher.seconds(),student_velocity_s=timed_student.seconds(),
            allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),
            task_samples=dict(self.sampler.counts),cache=stats,request_keys=keys)
        current=int(Path('/sys/fs/cgroup/memory.current').read_text());limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
        row['container_memory_bytes']=current
        row['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if current>limit*.8 or row['reserved']>torch.cuda.get_device_properties(0).total_memory*.85:
            raise RuntimeError('Resource envelope exceeded')
        return row
    def validate_math(self):
        batch,_,_,_,_=self.batch()
        assert tuple(batch['observation.state'].shape)==(self.args.batch,8)
        present=[key for key in self.teacher.config.image_features if key in batch]
        assert len(present)==2
        assert all(tuple(batch[key].shape)==(self.args.batch,3,256,256) for key in present)
        images,masks=self.teacher.prepare_images(batch);state=self.teacher.prepare_state(batch)
        missing=len(self.teacher.config.image_features)-len(present)
        expected_cameras=len(present)+min(missing,self.teacher.config.empty_cameras)
        resize=self.teacher.config.resize_imgs_with_padding
        internal_hw=tuple(resize) if resize is not None else (256,256)
        assert len(images)==len(masks)==expected_cameras
        assert all(tuple(x.shape)==(self.args.batch,3,*internal_hw) for x in images)
        assert all(tuple(mask.shape)==(self.args.batch,) for mask in masks)
        noise=torch.randn(self.args.batch,50,32,device='cuda');ctx=torch.zeros(self.args.batch,1,1,device='cuda')
        self.student.eval()
        with torch.no_grad():
            for policy,n in ((self.teacher,2),(self.student,1)):
                official=policy.model.sample_actions(images,masks,batch['observation.language.tokens'],
                    batch['observation.language.attention_mask'],state,noise=noise.clone())
                custom=euler_rollout(CachedSmolVLAVelocity(policy,batch),noise.clone(),ctx,num_steps=n,track_grad=False)[:,-1]
                torch.testing.assert_close(custom,official,rtol=1e-5,atol=1e-6)
            tv=CachedSmolVLAVelocity(self.teacher,batch)
            out=teacher_only_loss(shared_prefix_velocity(self.student,tv),tv,noise,ctx,self.config)
            torch.testing.assert_close(out.loss,1.5*out.action_loss,rtol=1e-5,atol=1e-6)
            repeated=teacher_only_loss(shared_prefix_velocity(self.student,tv),tv,noise,ctx,self.config)
            torch.testing.assert_close(out.teacher_trajectory,repeated.teacher_trajectory,rtol=0,atol=0)
        self.student.train(); self.student.model.vlm_with_expert.vlm.eval()
        return {'euler_parity':True,'loss_is_1_5_endpoint':True,'teacher_repeatable':True,
                'input_image_keys':present,'internal_image_shapes':[list(x.shape) for x in images]}
    def offline(self,path):
        self.close_prefetch();self.student.eval();values=[]
        with torch.no_grad():
            for item in self.manifest['offline_validation']:
                row=self.reader.read(item['task_index'],item['episode_index'],item['frame_index'])
                batch=processed_observation(row,self.processor)
                for seed in item['noise_seeds']:
                    noise=torch.randn(1,50,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
                    tv=CachedSmolVLAVelocity(self.teacher,batch)
                    out=teacher_only_loss(shared_prefix_velocity(self.student,tv),tv,noise,torch.zeros(1,1,1,device='cuda'),self.config)
                    values.append(dict(task=item['task_index'],loss=float(out.loss),endpoint=float(out.action_loss)))
        assert len(values)==1280
        atomic_json(path.with_name('offline_'+path.stem+'.json'),dict(checkpoint_sha256=sha256(path),
            pairs=values,mean_loss=sum(x['loss'] for x in values)/1280))
        self.student.train();self.student.model.vlm_with_expert.vlm.eval()


def probe(e,args):
    before=fingerprint(e.student,True); teacher_before=fingerprint(e.teacher)
    report=dict(passed=False,provenance=e.provenance,checks=e.validate_math())
    # Save/restore must replay consumed observations, RNG, optimizer and updates.
    e.step(); path=args.output/'recovery.pt';e.checkpoint(path)
    a=e.step(); expected=[p.detach().cpu().clone() for p in e.params]; sampler=copy.deepcopy(e.sampler.state_dict())
    e.restore(path); b=e.step()
    assert a['request_keys']==b['request_keys'] and sampler==e.sampler.state_dict()
    for p,q in zip(e.params,expected): torch.testing.assert_close(p.cpu(),q,rtol=1e-5,atol=1e-6)
    assert before==fingerprint(e.student,True) and teacher_before==fingerprint(e.teacher)
    report['checks'].update(restore_passed=True,frozen_unchanged=True)
    print('PREFLIGHT_PASSED',json.dumps(report['checks']),flush=True)
    for _ in range(10): e.step()
    windows=[]
    for _ in range(2):
        rows=[];start=time.perf_counter()
        while time.perf_counter()-start<60: rows.append(e.step())
        elapsed=time.perf_counter()-start
        windows.append(dict(samples_per_second=len(rows)*args.batch/elapsed,seconds=elapsed,updates=rows,
                            wait_fraction=sum(x['wait_s'] for x in rows)/elapsed))
        print('BENCHMARK_WINDOW',len(windows),len(rows)*args.batch/elapsed,flush=True)
    assert before==fingerprint(e.student,True) and teacher_before==fingerprint(e.teacher)
    report.update(passed=True,windows=windows,samples_per_second=sum(w['samples_per_second'] for w in windows)/2,
                  wait_fraction=sum(w['wait_fraction'] for w in windows)/2)
    atomic_json(args.output/'probe.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','data','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=['probe','train'],required=True)
    p.add_argument('--batch',type=int,choices=[16,32],required=True)
    p.add_argument('--decode-workers',type=int,choices=[1,2],default=1)
    p.add_argument('--resume',type=Path);p.add_argument('--gate',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'error.json').exists():raise RuntimeError('Existing error requires inspection')
    e=None;resources=ResourceLog(args.output)
    try:
        e=Engine(args)
        e.resources=resources
        if args.mode=='probe': probe(e,args);return
        if args.gate is None:raise ValueError('Preflight required')
        gate=json.loads(args.gate.read_text())
        if not gate['passed'] or gate['provenance']!=e.provenance:raise ValueError('Preflight mismatch')
        atomic_json(args.output/'provenance.json',e.provenance)
        if args.resume:e.restore(args.resume)
        # A training segment ends at the next prescribed closed-loop node.
        node=((e.update*args.batch)//40000+1)*40000
        if node>160000:raise ValueError('Budget already consumed')
        while e.update*args.batch<node:
            row=e.step()
            with (args.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if e.update%100==0:print(json.dumps({k:v for k,v in row.items() if k not in ('request_keys','task_samples','cache')}),flush=True)
            consumed=e.update*args.batch
            if consumed%16000==0 or consumed%40000==0:
                path=args.output/f'{e.update:06d}.pt';e.checkpoint(path);e.offline(path)
        print('CLOSED_LOOP_REQUIRED',str(path),flush=True)
    except BaseException:
        import traceback
        atomic_json(args.output/'error.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        if e:e.close_prefetch()
        resources.close()


if __name__=='__main__':main()
