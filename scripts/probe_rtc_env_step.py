"""Replay retained actions; attribute environment cost and test CPU affinity. No formal rows."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor


class Profile:
    def __init__(self):self.seconds=collections.defaultdict(float);self.calls=collections.Counter();self.restore=[]
    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        def measured(*args,**kwargs):
            start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:self.seconds[label]+=time.perf_counter()-start;self.calls[label]+=1
        setattr(obj,name,measured);self.restore.append((obj,name,original))
    def close(self):
        for obj,name,original in reversed(self.restore):setattr(obj,name,original)


def main():
    p=argparse.ArgumentParser()
    for name in ('output','checkpoint','student','assets-dir','manifest','error'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('New diagnostic directory required')
    import numpy as np
    import torch
    from scripts import run_libero_rollout
    from scripts.evaluate_s1_interleaved import make_env
    from scripts.s1_rtc_process import ProcessPolicy
    from smolvla_flow.benchmark40 import atomic_json,sha256
    from scripts.report_s1_rtc import distribution
    run_libero_rollout._configure_libero(args.assets_dir)
    import libero.libero as core
    from libero.libero import benchmark
    core._assets_path_cache=str(args.assets_dir)
    manifest=json.loads(args.manifest.read_text());error=json.loads(args.error.read_text())
    if sha256(args.student)!=manifest['student_sha256']:raise ValueError('Student identity')
    task=next(t for t in manifest['tasks'] if (t['suite'],t['task_id'])==(error['suite'],error['task_id']))
    actions=[t['action'] for t in error['ticks']]
    if len(actions)!=94:raise ValueError('Expected the retained 94-tick failure')
    request_ticks={q['observed_tick'] for q in error['requests'] if not q['initial']}
    allowed=set(os.sched_getaffinity(0));unique=[];seen=set()
    for cpu in sorted(allowed):
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=(root.joinpath('physical_package_id').read_text(),root.joinpath('core_id').read_text())
        if key not in seen:unique.append(cpu);seen.add(key)
    if len(unique)<6:raise ValueError('Insufficient distinct eligible CPU cores')
    env_cpus=set(unique[:2]);policy_cpus=set(unique[2:6])
    def affinity(pid,cpus):
        for tid in Path(f'/proc/{pid}/task').iterdir():
            try:os.sched_setaffinity(int(tid.name),cpus)
            except ProcessLookupError:pass
    torch.set_num_threads(1);policy=ProcessPolicy(args)
    args.output.mkdir(parents=True);results=[];reference=None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for condition in ('baseline','affinity'):
                affinity(os.getpid(),allowed if condition=='baseline' else env_cpus)
                affinity(policy.process.pid,allowed if condition=='baseline' else policy_cpus)
                for repeat in range(3):
                    env=make_env(benchmark.get_benchmark_dict()[task['suite']](),task);profile=Profile()
                    try:
                        env.init_state_id=error['init_index'];obs,_=env.reset(seed=123);policy.reset()
                        base=env._env
                        while 'sim' not in vars(base):
                            base=vars(base).get('env',vars(base).get('_env'))
                            if base is None:raise RuntimeError('Cannot resolve raw environment')
                        sim=base.sim;context=sim._render_context_offscreen
                        for obj,name,label in [(sim,'step','physics_step'),(sim,'forward','physics_forward'),
                            (base,'_pre_action','robot_control'),(base,'_update_observables','observables_inclusive'),
                            (base,'_get_observations','observation_assembly'),(base,'_post_action','post_action'),
                            (context,'render','render'),(context,'read_pixels','read_pixels'),
                            (env,'_format_raw_obs','format_observation')]:profile.wrap(obj,name,label)
                        policy.predict(obs,task['language'],error['mode'],error['delay_ms'])
                        rows=[];future=None;predictions=[];digest=hashlib.sha256();start=time.perf_counter()
                        def hash_obs(x):
                            if isinstance(x,dict):
                                for k in sorted(x):digest.update(k.encode());hash_obs(x[k])
                            else:
                                a=np.asarray(x);digest.update(str(a.shape).encode());digest.update(a.tobytes())
                        for tick,action in enumerate(actions):
                            deadline=start+tick/20;time.sleep(max(0.,deadline-time.perf_counter()))
                            if tick in request_ticks:
                                if future is not None:predictions.append(future.result(timeout=30)['metrics'])
                                future=pool.submit(policy.predict,obs,task['language'],error['mode'],error['delay_ms'])
                            before=time.perf_counter();late=max(0.,before-deadline)
                            profile.seconds.clear();profile.calls.clear()
                            obs,_,_,_,info=env.step(np.asarray(action,dtype=np.float32))
                            elapsed=time.perf_counter()-before
                            parts=dict(profile.seconds);parts['observable_nonrender']=max(0.,parts.get('observables_inclusive',0)-parts.get('render',0)-parts.get('read_pixels',0))
                            rows.append(dict(tick=tick,step_s=elapsed,lateness_s=late,parts=parts,calls=dict(profile.calls),success=bool(info.get('is_success',False))))
                            hash_obs(obs)
                        if future is not None:predictions.append(future.result(timeout=30)['metrics'])
                        obs_hash=digest.hexdigest()
                        if reference is None:reference=obs_hash
                        result=dict(condition=condition,repeat=repeat,observation_sha256=obs_hash,
                            observation_equal=obs_hash==reference,late_fraction=sum(r['lateness_s']>.005 for r in rows)/94,
                            step=distribution(r['step_s'] for r in rows),
                            parts={k:distribution(r['parts'].get(k,0) for r in rows) for k in rows[0]['parts']},
                            ticks=rows,requests=predictions)
                        results.append(result);atomic_json(args.output/f'{condition}_{repeat}.json',result)
                        print(json.dumps({k:v for k,v in result.items() if k not in ('ticks','requests')}),flush=True)
                    finally:profile.close();env.close()
        atomic_json(args.output/'summary.json',dict(results=[{k:v for k,v in r.items() if k not in ('ticks','requests')} for r in results],
            environment_cpus=sorted(env_cpus),policy_cpus=sorted(policy_cpus),allowed_cpus=sorted(allowed),
            cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),error_sha256=sha256(args.error),
            entry_sha256=sha256(Path(__file__)),limitation='Instrumented fixed-action replay, not a formal success evaluation. Nested timers are not additive. Host CPUs are not exclusively reserved.'))
    finally:policy.close();affinity(os.getpid(),allowed)


if __name__=='__main__':
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    main()
