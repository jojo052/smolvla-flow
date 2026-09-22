#!/usr/bin/env python3
"""T1: independent CPU environments and one strictly batch=1 GPU server.

check-and-run first compares four native serial pilots against interleaved
pilots, then starts formal episodes only if the fixed numerical gate passes.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import random
from pathlib import Path
import threading
import time
import traceback
import types

import numpy as np
import torch
from scripts.evaluate_distill40 import frozen
from scripts.run_libero_rollout import _load_policy, parse_args, _make_observation_pipeline
from smolvla_flow.async_runtime import LeRobotPostprocessorAdapter
from smolvla_flow.benchmark40 import atomic_json, freeze_manifest, sha256, episode_key, validate_episode


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def env_class():
    from lerobot.envs.libero import LiberoEnv
    class Env(LiberoEnv):
        def step(self, action):
            raw, reward, done, info = self._env.step(action)
            success = bool(self._env.check_success())
            return self._format_raw_obs(raw), reward, bool(done or success), False, {**info, 'is_success': success}
    return Env


def make_env(suite, task):
    return env_class()(task_suite=suite, task_id=task['task_id'], task_suite_name=task['suite'],
        episode_length=task['max_steps'], obs_type='pixels_agent_pos', observation_width=256,
        observation_height=256, num_steps_wait=10, init_states=True, n_envs=1, control_mode='relative')


def worker(connection, assets, output, run_id, phase):
    """One bounded request in flight; no policy tensors or global CUDA RNG."""
    torch.set_num_threads(1)
    try:
        from scripts.run_libero_rollout import _configure_libero
        _, actual_assets = _configure_libero(Path(assets))
        import libero.libero as core
        core._assets_path_cache = str(actual_assets)
        from libero.libero import benchmark
        import imageio.v2 as imageio
        while True:
            job = connection.recv()
            if job is None:
                break
            task, indices = job
            suite = benchmark.get_benchmark_dict()[task['suite']]()
            env = make_env(suite, task)
            try:
                for index in indices:
                    key = episode_key('T1', task['suite'], task['task_id'], index, phase)
                    directory = Path(output)/Path(key).parent
                    directory.mkdir(parents=True, exist_ok=True)
                    total_start = time.perf_counter()
                    random.seed(123); np.random.seed(123); torch.manual_seed(123)
                    env.init_state_id = index
                    obs, _ = env.reset(seed=123)
                    assert env.init_state_id == index+1 and env.num_steps_wait == 10
                    reset_s = time.perf_counter()-total_start
                    frames, predictions, selects, waits, noises, actions = [], [], [], [], [], []
                    record = any(not (directory/f'first_{k}.mp4').exists() for k in ('success','failure'))
                    start = time.perf_counter()
                    reward_sum, action_max, success = 0., 0., False
                    for step in range(task['max_steps']):
                        if record:
                            frames.append(np.ascontiguousarray(obs['pixels']['image'][::-1, ::-1]))
                        sent = time.perf_counter()
                        connection.send(('predict', key, step, task['language'], obs, sent))
                        reply_key, reply_step, action, metrics = connection.recv()
                        if (reply_key, reply_step) != (key, step):
                            raise ValueError('Request/response identity mismatch')
                        if action.shape != (7,) or not np.isfinite(action).all():
                            raise ValueError('Nonfinite action')
                        predictions.append(metrics['prediction_s']); selects.append(metrics['select_s'])
                        waits.append(metrics['queue_s']); noises.append(metrics['noise_sha256'])
                        if phase == 'pilot': actions.append(metrics['normalized'])
                        action_max = max(action_max, float(np.abs(action).max()))
                        obs, reward, terminated, truncated, info = env.step(action)
                        reward_sum += float(reward)
                        success = bool(info['is_success'])
                        if success: break
                        if terminated or truncated:
                            raise RuntimeError('Unexpected early termination')
                    elapsed, total = time.perf_counter()-start, time.perf_counter()-total_start
                    kind = 'success' if success else 'failure'
                    video = directory/f'first_{kind}.mp4'
                    created = False
                    if record and not video.exists():
                        frames.append(np.ascontiguousarray(obs['pixels']['image'][::-1, ::-1]))
                        temporary = video.with_name(video.stem+'.tmp.mp4')
                        imageio.mimwrite(temporary, frames, fps=20, codec='libx264', quality=6)
                        temporary.replace(video); created = True
                    row = dict(status='complete', run_id=run_id, phase=phase, model='T1', key=key,
                        suite=task['suite'], task_id=task['task_id'], task_name=task['name'],
                        language=task['language'], init_index=index, init_state_sha256=task['state_sha256'][index],
                        policy_seed=123, environment_seed=123, settle_steps=10, steps=len(predictions),
                        success=success, failure_type=None if success else 'timeout', reward_sum=reward_sum,
                        elapsed_seconds=elapsed, total_seconds=total, reset_seconds=reset_s,
                        prediction_seconds=predictions, select_seconds=selects, request_wait_seconds=waits,
                        noise_sha256=noises, action_nonfinite_count=0, action_abs_max=action_max,
                        video=str(video) if created else None, video_created_for_this_episode=created,
                        action_processing='checkpoint_unnormalization_only_no_extra_hysteresis')
                    if phase == 'pilot': row['normalized_actions'] = actions
                    connection.send(('complete', key, row))
                    if connection.recv() != ('saved', key): raise RuntimeError('Missing durable write acknowledgement')
            finally:
                env.close()
            connection.send(('idle',))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


class Resources:
    def __init__(self, output):
        self.stop = threading.Event(); self.samples=[]; self.error=None; self.output=output
        self.thread = threading.Thread(target=self.loop, daemon=True)
    def loop(self):
        import subprocess
        while not self.stop.is_set():
            try:
                util, used, total = map(float, subprocess.check_output(['nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip().split(','))
                current=int(Path('/sys/fs/cgroup/memory.current').read_text())
                limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
                sample=dict(time=time.time(),gpu_util=util,vram_mib=used,vram_total_mib=total,
                            container_memory_bytes=current,container_limit_bytes=limit)
                self.samples.append(sample)
                if used/total > .85 or current/limit > .8:
                    self.error='Resource safety limit exceeded'
            except Exception:
                self.error=traceback.format_exc()
            self.stop.wait(5)
    def __enter__(self): self.thread.start(); return self
    def __exit__(self,*_):
        self.stop.set(); self.thread.join(timeout=10)
        atomic_json(self.output/'resources.json',dict(samples=self.samples,error=self.error))


def run_parallel(args, manifest, loaded, output, phase, workers):
    output=Path(output)
    run_manifest=dict(manifest,execution=dict(workers=workers,batch=1,phase=phase,entry_sha256=sha256(Path(__file__))))
    run_id=freeze_manifest(output/'manifest.json',run_manifest)
    jobs=[]; rows=[]
    for task in manifest['tasks']:
        if phase=='pilot' and task['task_id']!=0: continue
        pending=[]
        for index in ([10] if phase=='pilot' else range(10)):
            key=episode_key('T1',task['suite'],task['task_id'],index,phase)
            path=output/(key+'.json')
            if path.exists():
                row=json.loads(path.read_text()); validate_episode(row,run_id,key)
                if row['init_state_sha256']!=task['state_sha256'][index]: raise ValueError('Resume state mismatch')
                if row['checkpoint_sha256']!=manifest['checkpoint_files']['model.safetensors']: raise ValueError('Resume weight mismatch')
                rows.append(row)
            else: pending.append(index)
        if pending: jobs.append((task,pending))
    policy=loaded.policy; policy.reset()
    generators={}; next_step={}; prepares={}; context=mp.get_context('spawn')
    processes=[]; connections=[]; busy=set()
    for _ in range(min(workers,len(jobs))):
        parent,child=context.Pipe()
        proc=context.Process(target=worker,args=(child,str(args.assets_dir),str(output),run_id,phase))
        proc.start(); child.close(); processes.append(proc); connections.append(parent)
    def assign(conn):
        if jobs: conn.send(jobs.pop(0)); busy.add(conn)
        else: conn.send(None)
    for conn in connections: assign(conn)
    post=LeRobotPostprocessorAdapter(loaded.postprocessor)
    started=time.perf_counter()
    try:
        with Resources(output) as resources:
            while busy:
                if resources.error: raise RuntimeError(resources.error)
                ready=wait(list(busy),timeout=5)
                if not ready and any(not p.is_alive() and p.exitcode for p in processes):
                    raise RuntimeError('Environment worker exited unexpectedly')
                for conn in ready:
                    msg=conn.recv(); kind=msg[0]
                    if kind=='error': raise RuntimeError(msg[1])
                    if kind=='idle': busy.remove(conn); assign(conn); continue
                    if kind=='complete':
                        _,key,row=msg
                        if next_step.pop(key)!=row['steps']: raise ValueError('Step count mismatch')
                        generators.pop(key)
                        row['checkpoint_sha256']=manifest['checkpoint_files']['model.safetensors']
                        validate_episode(row,run_id,key)
                        atomic_json(output/(key+'.json'),row); rows.append(row)
                        conn.send(('saved',key))
                        print('COMPLETE',len(rows),key,row['success'],flush=True)
                        continue
                    if kind!='predict': raise ValueError('Unknown worker message')
                    _,key,step,language,obs,sent=msg
                    if step==0:
                        if key in generators: raise ValueError('Duplicate episode')
                        generators[key]=torch.Generator(device='cuda').manual_seed(123); next_step[key]=0
                    if next_step[key]!=step: raise ValueError('Nonsequential request')
                    queue_s=time.perf_counter()-sent
                    if language not in prepares: prepares[language]=_make_observation_pipeline(language)
                    batch=prepares[language](obs,loaded.preprocessor)
                    if tuple(batch['observation.state'].shape)!=(1,8): raise ValueError('State contract')
                    images=[v for k,v in batch.items() if k.startswith('observation.images.')]
                    if len(images)!=2 or any(tuple(x.shape)!=(1,3,256,256) for x in images): raise ValueError('Image contract')
                    noise=torch.normal(0.,1.,size=(1,50,32),device='cuda',generator=generators[key])
                    noise_hash=digest(noise)
                    policy.reset()  # This interface does not carry queues across requests.
                    torch.cuda.synchronize(); begin=time.perf_counter()
                    with torch.inference_mode(): chunk=policy._get_action_chunk(batch,noise=noise)
                    torch.cuda.synchronize(); prediction_s=time.perf_counter()-begin
                    if tuple(chunk.shape)!=(1,50,7) or not torch.isfinite(chunk).all(): raise ValueError('Chunk contract')
                    normalized=chunk[0,0]
                    action=post(normalized).numpy().astype(np.float32)
                    metrics=dict(prediction_s=prediction_s,select_s=time.perf_counter()-begin,queue_s=queue_s,
                                 noise_sha256=noise_hash,normalized=normalized.cpu().tolist())
                    conn.send((key,step,action,metrics)); next_step[key]+=1
        expected=4 if phase=='pilot' else 400
        if len(rows)!=expected: raise ValueError('Incomplete episode set')
        atomic_json(output/'result.json',dict(complete=True,episodes=rows,successes=sum(r['success'] for r in rows),
            end_to_end_seconds=time.perf_counter()-started,efficiency_mode='interleaved_batch1_not_historical_serial'))
        return rows
    except BaseException:
        atomic_json(output/'error.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
    finally:
        for conn in connections: conn.close()
        for proc in processes:
            proc.join(timeout=2)
            if proc.is_alive(): proc.terminate(); proc.join(timeout=5)


def serial_pilots(args,manifest,suites,loaded):
    output=args.output/'serial_pilot'; output.mkdir(parents=True,exist_ok=True)
    if args.reference:
        previous=json.loads((args.reference/'manifest.json').read_text())
        for field in ('protocol','checkpoint_files','tasks','sources','versions','assets_sha256'):
            if previous[field]!=manifest[field]: raise ValueError('Serial reference protocol changed: '+field)
        rows=[json.loads(p.read_text()) for p in sorted((args.reference/'serial_pilot').glob('*.json'))]
        if len(rows)!=4 or len({r['suite'] for r in rows})!=4: raise ValueError('Incomplete reference')
        for row in rows:
            if row['init_index']!=10 or row['task_id']!=0 or row['policy_seed']!=123:
                raise ValueError('Reference identity mismatch')
        atomic_json(output/'reused_reference.json',dict(directory=str(args.reference),
            files={p.name:sha256(p) for p in sorted((args.reference/'serial_pilot').glob('*.json'))}))
        return rows
    copy=dict(frozen.run_episode.__globals__)
    copy['episode_key']=lambda _m,s,t,i,p:episode_key('T1',s,t,i,p)
    rollout=types.FunctionType(frozen.run_episode.__code__,copy)
    rows=[]
    for task in manifest['tasks']:
        if task['task_id']!=0: continue
        path=output/f"{task['suite']}.json"
        if path.exists(): raise ValueError('Serial pilot already exists; use a fresh check directory')
        env=make_env(suites[task['suite']],task)
        noises=[]; actions=[]
        native_noise=loaded.policy.model.sample_noise; native_select=loaded.policy.select_action
        def record_noise(*a,**kw):
            noise=native_noise(*a,**kw); noises.append(digest(noise)); return noise
        def record_action(*a,**kw):
            value=native_select(*a,**kw); actions.append(value[0].detach().cpu().tolist()); return value
        loaded.policy.model.sample_noise=record_noise; loaded.policy.select_action=record_action
        try:
            row=rollout(env,loaded,task,10,'pilot','serial-reference',
                        {k:output/f"{task['suite']}_{k}.mp4" for k in ('success','failure')})
            row.update(model='T1',noise_sha256=noises,normalized_actions=actions)
            atomic_json(path,row); rows.append(row)
            print('SERIAL_PILOT',task['suite'],row['steps'],row['success'],flush=True)
        finally:
            loaded.policy.model.sample_noise=native_noise; loaded.policy.select_action=native_select; env.close()
    return rows


def compare_pilots(serial,parallel):
    if len(serial)!=4 or len(parallel)!=4 or len({r['suite'] for r in serial})!=4 or len({r['suite'] for r in parallel})!=4:
        return dict(passed=False,results=[],reason='Expected four distinct suites per execution')
    result=[]
    for a,b in zip(sorted(serial,key=lambda r:r['suite']),sorted(parallel,key=lambda r:r['suite'])):
        x=np.array(a['normalized_actions']); y=np.array(b['normalized_actions'])
        aligned=(a['suite'],a['init_state_sha256'],a['steps'],a['success'])==(b['suite'],b['init_state_sha256'],b['steps'],b['success'])
        same_noise=a['noise_sha256']==b['noise_sha256']
        close=x.shape==y.shape and np.allclose(y,x,rtol=1e-5,atol=1e-6)
        result.append(dict(suite=a['suite'],outcome_aligned=aligned,noise_equal=same_noise,
                           actions_close=bool(close),max_abs=float(np.max(np.abs(y-x))) if x.shape==y.shape else None))
    return dict(passed=all(r['outcome_aligned'] and r['noise_equal'] and r['actions_close'] for r in result),results=result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--assets-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--mode',choices=['check','check-and-run','resume'],default='check')
    p.add_argument('--reference',type=Path,help='Reuse unchanged native serial pilots after execution-only fixes')
    args=p.parse_args()
    if list(args.output.rglob('error.json')): raise RuntimeError('Recorded execution error requires inspection')
    quota,period=map(int,Path('/sys/fs/cgroup/cpu.max').read_text().split())
    if not 1<=args.workers<=min(16,quota//period-4): raise ValueError('Worker CPU quota violation')
    torch.set_num_threads(4)
    config=json.loads((frozen.ROOT/'configs/libero40_public_v1.json').read_text())
    manifest,suites=frozen.prepare_manifest(args,config)
    manifest['protocol']=dict(manifest['protocol'],flow_steps=1)
    manifest['new_entry_sha256']=sha256(Path(__file__))
    freeze_manifest(args.output/'manifest.json',manifest)
    loaded=_load_policy(parse_args(['--checkpoint',str(args.checkpoint),'--flow-steps','1','--disable-rtc']),action_execution_steps=1)
    loaded.policy.eval()
    for parameter in loaded.policy.parameters(): parameter.requires_grad_(False)
    if args.mode!='resume':
        serial=serial_pilots(args,manifest,suites,loaded)
        parallel=run_parallel(args,manifest,loaded,args.output/'parallel_pilot','pilot',args.workers)
        gate=compare_pilots(serial,parallel)
        gate.update(entry_sha256=sha256(Path(__file__)),workers=args.workers,rtol=1e-5,atol=1e-6)
        atomic_json(args.output/'pilot_gate.json',gate)
        if not gate['passed']: raise RuntimeError('Pilot consistency gate failed; formal evaluation not started')
    gate=json.loads((args.output/'pilot_gate.json').read_text())
    if not gate['passed'] or gate['entry_sha256']!=sha256(Path(__file__)) or gate['workers']!=args.workers:
        raise ValueError('Missing/mismatched pilot authorization')
    if args.mode in ('check-and-run','resume'):
        run_parallel(args,manifest,loaded,args.output/'formal_run','formal',args.workers)


if __name__=='__main__':
    os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
    main()
