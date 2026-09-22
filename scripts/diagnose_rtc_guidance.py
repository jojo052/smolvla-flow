"""Isolated fixed-input RTC strength diagnosis. Never changes formal results."""
import argparse
import copy
import json
import os
from pathlib import Path

WEIGHTS = (0., .1, .5, 1., 2., 10.)


def main():
    p = argparse.ArgumentParser()
    for key in ('output', 'checkpoint', 'student', 'assets-dir', 'source'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new diagnostic output directory')
    import numpy as np
    import torch
    from scripts import run_libero_rollout as loader
    from scripts.evaluate_s1_interleaved import make_env
    from scripts.distill_smolvla_action_expert import _configure_action_only_student
    from scripts.train_distill40_stage import overlay
    from smolvla_flow.async_runtime import LeRobotPostprocessorAdapter
    from smolvla_flow.benchmark40 import atomic_json, sha256
    from scripts.evaluate_s1_rtc import STUDENT_SHA, state_hash
    manifest = json.loads((args.source/'manifest.json').read_text())
    if sha256(args.student) != STUDENT_SHA:
        raise ValueError('S1 identity mismatch')
    for path, digest in manifest['entry_sources'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Source drift: '+path)
    args.output.mkdir(parents=True)
    atomic_json(args.output/'design.json', dict(weights=WEIGHTS, delays=[3,5], seeds=[123,124],
        tasks='four suite task0 init13', snapshots=[10,30,60], prefix='shared S1 prediction ten ticks earlier',
        selection='largest weight <=1 with every case weighted error nonincreasing and normalized amplitude within plain/prefix envelope',
        source_manifest_sha256=sha256(args.source/'manifest.json'), student_sha256=STUDENT_SHA,
        entry_sha256=sha256(Path(__file__)), purpose='diagnostic only, no formal success rates'))
    loader._configure_libero(args.assets_dir)
    import libero.libero as core
    from libero.libero import benchmark
    core._assets_path_cache = str(args.assets_dir)
    tasks = [t for t in manifest['tasks'] if t['task_id']==0]
    samples = []
    torch.set_num_threads(1)
    for task in tasks:
        source = args.source/'episodes/check'/task['suite']/'task00/init13/async_d0.json'
        row = json.loads(source.read_text())
        suite = benchmark.get_benchmark_dict()[task['suite']]()
        if state_hash(suite.get_task_init_states(0)[13]) != task['state_sha256'][13]:
            raise ValueError('Initial state drift')
        env = make_env(suite, task)
        try:
            env.init_state_id=13
            obs,_=env.reset(seed=123)
            snapshots={0:copy.deepcopy(obs)}
            for i,tick in enumerate(row['ticks'][:60],1):
                obs,*_=env.step(np.asarray(tick['action'],dtype=np.float32))
                if i in (10,20,30,50,60): snapshots[i]=copy.deepcopy(obs)
            for tick in (10,30,60):
                if tick in snapshots and tick-10 in snapshots:
                    samples.append(dict(task=task, tick=tick, previous=snapshots[tick-10], observation=snapshots[tick],
                                        replay_sha256=sha256(source)))
        finally: env.close()
    torch.save(samples,args.output/'observations.pt')
    rows=[]; stored=[]
    for name,steps in (('S1',1),('T2',2)):
        torch.set_num_threads(4)
        loaded=loader._load_policy(loader.parse_args(['--checkpoint',str(args.checkpoint),'--flow-steps',str(steps),
            '--mode','async','--rtc-max-guidance-weight','10','--rtc-schedule','exp']),action_execution_steps=10)
        policy=loaded.policy
        if name=='S1': overlay(policy,args.student,_configure_action_only_student(policy))
        policy.eval()
        for parameter in policy.parameters(): parameter.requires_grad_(False)
        model=policy.model; rtc=model.rtc_processor
        post=LeRobotPostprocessorAdapter(loaded.postprocessor)
        original=model.denoise_step; times=[]
        def counted(*a,**kw):
            times.append(kw['timestep'].detach().cpu().tolist())
            return original(*a,**kw)
        model.denoise_step=counted
        def predict(batch,noise,prefix=None,delay=0,weight=0.):
            times.clear();enabled=prefix is not None and weight>0
            model.rtc_processor=rtc if enabled else None
            for cfg in (policy.config,model.config): cfg.rtc_config.enabled=enabled
            rtc.rtc_config.max_guidance_weight=weight
            with torch.enable_grad():
                out=policy._get_action_chunk(batch,noise=noise.clone(),prev_chunk_left_over=prefix,
                    inference_delay=delay,execution_horizon=10).detach()
            if len(times)!=steps or any(abs(t[0]-(1-i/steps))>1e-6 for i,t in enumerate(times)):
                raise ValueError('Integration count/time drift')
            if not torch.isfinite(out).all():raise ValueError('Nonfinite output')
            return out
        for si,sample in enumerate(samples):
            prepare=loader._make_observation_pipeline(sample['task']['language'])
            batch=prepare(sample['observation'],loaded.preprocessor)
            previous_batch=prepare(sample['previous'],loaded.preprocessor)
            for seed in (123,124):
                generator=torch.Generator(device='cuda').manual_seed(seed)
                previous_noise=torch.randn((1,50,32),device='cuda',generator=generator)
                noise=torch.randn((1,50,32),device='cuda',generator=generator)
                index=si*2+(seed-123)
                if name=='S1':
                    prefix=predict(previous_batch,previous_noise)[:,10:]
                    stored.append(dict(noise=noise.cpu(),previous_noise=previous_noise.cpu(),prefix=prefix.cpu()))
                else: prefix=stored[index]['prefix'].cuda()
                plain=predict(batch,noise)
                for delay in (3,5):
                    weights=rtc.get_prefix_weights(delay,10,50)[:40].to(plain.device)[None,:,None]
                    base_error=((plain[:,:40]-prefix).square()*weights).mean().item()
                    envelope=max(plain.abs().max().item(),prefix.abs().max().item())
                    for weight in WEIGHTS:
                        out=predict(batch,noise,prefix,delay,weight)
                        actions=np.asarray([post(a).tolist() for a in out[0]])
                        err=((out[:,:40]-prefix).square()*weights).mean().item()
                        metric=dict(model=name,suite=sample['task']['suite'],tick=sample['tick'],seed=seed,
                            delay=delay,weight=weight,weighted_error=err,plain_error=base_error,
                            normalized_max=out.abs().max().item(),envelope=envelope,
                            executed_motion_over1=float((np.abs(actions[delay:,:6])>1).any(axis=1).mean()),
                            max_correction=(out-plain).abs().max().item(),
                            finite=True,gate=err<=base_error+1e-6 and out.abs().max().item()<=envelope+1e-5)
                        rows.append(metric)
                        np.save(args.output/f'{name}_{si}_{seed}_{delay}_{weight}.npy',out.cpu().numpy())
            print('DONE',name,si,flush=True)
        del model,rtc,policy,loaded,original,batch,previous_batch
        import gc
        gc.collect();torch.cuda.empty_cache()
    torch.save(stored,args.output/'noise_prefix.pt')
    passing=[w for w in WEIGHTS if 0<w<=1 and all(r['gate'] for r in rows if r['model']=='S1' and r['weight']==w)]
    atomic_json(args.output/'summary.json',dict(rows=rows,selected_weight=max(passing) if passing else None,
        observations=len(samples),observation_sha256=sha256(args.output/'observations.pt'),
        noise_prefix_sha256=sha256(args.output/'noise_prefix.pt')))
    print('SELECTED',max(passing) if passing else None,flush=True)


if __name__=='__main__':
    os.environ.setdefault('MUJOCO_GL','egl')
    main()
