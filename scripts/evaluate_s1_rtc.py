"""Independent 20 Hz, single-environment S1 RTC benchmark. No training or power control."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from smolvla_flow.benchmark40 import atomic_json,freeze_manifest,sha256
from smolvla_flow.rtc_benchmark import PROTOCOL,SUITES,Timeline,schedule,validate_row,wait_boundary,accept_recorded

STUDENT_SHA='6397b43bcecea024ff68b03a2fcedbf81d726f98fa8dbecb7b08187097dfb9aa'


def state_hash(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def relative_control_evidence(env):
    """Do not assume that repeating a delta is holding the end effector still."""
    seen=set();todo=[env];evidence=[]
    while todo:
        obj=todo.pop()
        if id(obj) in seen:continue
        seen.add(id(obj))
        for robot in getattr(obj,'robots',[]):
            controllers=getattr(robot,'part_controllers',None)
            if controllers is None:controllers={'arm':getattr(robot,'controller',None)}
            for name,ctrl in controllers.items():
                if ctrl is not None and hasattr(ctrl,'use_delta'):
                    evidence.append(dict(name=name,use_delta=bool(ctrl.use_delta),type=type(ctrl).__name__))
        for name in ('env','_env'):
            child=getattr(obj,name,None)
            if child is not None and child is not obj:todo.append(child)
    if not evidence or not all(x['use_delta'] for x in evidence):
        raise RuntimeError('Relative controller semantics could not be verified')
    return evidence


def episode(args,job,task,suite,policy,run_id,resources):
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from scripts.evaluate_s1_interleaved import make_env
    wall_start=time.time();start=time.perf_counter();env=make_env(suite,task);pool=ThreadPoolExecutor(max_workers=1)
    future=None;request_id=None;row=None
    try:
        random.seed(123);np.random.seed(123);torch.manual_seed(123);policy.reset()
        env.init_state_id=job['init_index'];obs,_=env.reset(seed=123)
        evidence=relative_control_evidence(env)
        if env.init_state_id!=job['init_index']+1 or env.num_steps_wait!=10:raise ValueError('Initial-state contract')
        frames=[];ticks=[];requests=[];success=False;previous_action=None;previous_sequence=None
        directory=args.output/'videos'/job['phase']/job['suite']/f"task{job['task_id']:02d}"/f"{job['mode']}_d{job['delay_ms']}"
        record=any(not (directory/f'first_{kind}.mp4').exists() for kind in ('success','failure'))
        initial=policy.predict(copy.deepcopy(obs),task['language'],job['mode'],job['delay_ms'])
        timeline=Timeline(job['mode']);timeline.prime(initial['normalized'],initial['actions'],initial['metrics']['elapsed_s'])
        requests.append(dict(request_id=0,observed_tick=0,accepted_tick=0,actual_delay=0,initial=True,**initial['metrics']))
        control_start=time.perf_counter()
        for tick in range(task['max_steps']):
            lateness=wait_boundary(control_start,tick,time.perf_counter,time.sleep)
            resources.check()
            # A response cannot enter the action queue between two control ticks.
            if future is not None and future.done():
                result=future.result()
                actual=accept_recorded(timeline,requests,request_id,tick,result)
                future=None
            if timeline.should_request():
                request_id,prefix,estimate=timeline.submit(tick)
                future=pool.submit(policy.predict,copy.deepcopy(obs),task['language'],job['mode'],job['delay_ms'],
                                   prefix if job['mode']=='rtc' else None,estimate)
            action,waiting,age=timeline.pop(tick)
            boundary=previous_sequence is not None and timeline.active_sequence!=previous_sequence and not waiting
            jump=None
            if boundary and previous_action is not None:
                difference=np.asarray(action)-np.asarray(previous_action)
                jump=dict(translation=float(np.linalg.norm(difference[:3])),
                          rotation=float(np.linalg.norm(difference[3:6])),gripper=float(abs(difference[6])))
            previous_action=action;previous_sequence=timeline.active_sequence
            if record:frames.append(np.ascontiguousarray(obs['pixels']['image'][::-1,::-1]))
            env_started=time.perf_counter()
            lateness=max(lateness,env_started-(control_start+tick/20))
            obs,reward,terminated,truncated,info=env.step(np.asarray(action,dtype=np.float32))
            ticks.append(dict(tick=tick,lateness_s=lateness,waiting=waiting,observation_age_ticks=age,
                action=action,chunk_boundary_jump=jump,environment_step_s=time.perf_counter()-env_started))
            success=bool(info.get('is_success',False))
            if success:break
            if (terminated or truncated) and tick+1<task['max_steps']:raise RuntimeError('Unexpected early termination')
        # Include the final 50 ms control interval in the controller duration.
        wait_boundary(control_start,len(ticks),time.perf_counter,time.sleep)
        control_seconds=time.perf_counter()-control_start
        if future is not None:
            result=future.result(timeout=30)
            requests.append(dict(request_id=request_id,accepted_tick=None,observed_tick=timeline.pending[1],
                actual_delay=None,initial=False,discarded_after_episode=True,**result['metrics']))
        late_fraction=sum(x['lateness_s']>.005 for x in ticks)/len(ticks)
        if late_fraction>.05:raise RuntimeError(f'20Hz deadline gate failed: {late_fraction:.3%} late ticks')
        video=None;kind='success' if success else 'failure'
        if record and not (directory/f'first_{kind}.mp4').exists():
            frames.append(np.ascontiguousarray(obs['pixels']['image'][::-1,::-1]))
            directory.mkdir(parents=True,exist_ok=True);target=directory/f'first_{kind}.mp4'
            temporary=target.with_name(target.stem+'.partial.mp4')
            imageio.mimwrite(temporary,frames,fps=20,codec='libx264',quality=6);temporary.replace(target)
            video=str(target.relative_to(args.output))
        row=dict(job,run_id=run_id,status='complete',success=success,steps=len(ticks),ticks=ticks,
            requests=requests,total_seconds=time.perf_counter()-start,control_seconds=control_seconds,
            late_fraction=late_fraction,video=video,init_state_sha256=task['state_sha256'][job['init_index']],
            student_sha256=STUDENT_SHA,controller_evidence=evidence)
        row.update(started_unix=wall_start,finished_unix=time.time())
        validate_row(row,job,run_id)
        return row
    except BaseException:
        atomic_json(args.output/'errors'/(job['key']+f'-{time.time_ns()}.json'),
            dict(job,run_id=run_id,status='execution_error',traceback=traceback.format_exc(),
                 ticks=locals().get('ticks',[]),requests=locals().get('requests',[])))
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True);env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('check','run','resume','report'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    for name in ('checkpoint','student','assets-dir','baseline-manifest','protocol'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--execution-mode',choices=('all','sync','async','rtc'),default='all')
    parser.add_argument('--delay-ms',choices=('all','0','100'),default='all')
    parser.add_argument('--split-process',action='store_true',help='Use a single policy subprocess, retaining the same controller protocol')
    parser.add_argument('--acknowledge-errors',action='store_true',help='After inspection, archive the error marker before retrying; never deletes evidence')
    args=parser.parse_args()
    if args.stage=='report':
        from scripts.report_s1_rtc import report
        report(args.output);return
    if any(getattr(args,k) is None for k in ('checkpoint','student','assets_dir','baseline_manifest','protocol')):
        parser.error('GPU stages require checkpoint, student, assets-dir, baseline-manifest and protocol')
    if json.loads(args.protocol.read_text())!=PROTOCOL:raise ValueError('Protocol differs from approved experiment')
    if args.stage=='check' and (args.execution_mode!='all' or args.delay_ms!='all'):
        raise ValueError('All 24 check episodes are mandatory')
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    error=args.output/'error.json'
    if error.exists():
        if not args.acknowledge_errors:raise RuntimeError('Recorded error requires inspection')
        archive=args.output/'errors'/f'acknowledged-{time.time_ns()}.json'
        archive.parent.mkdir(parents=True,exist_ok=True);error.replace(archive)
    resources=None;policy=None
    try:
        import torch
        from scripts.s1_rtc_policy import Policy
        from scripts import run_libero_rollout  # cache the current loader before path changes
        # Import runtime before frozen evaluator mutates package lookup paths.
        from scripts.evaluate_s1_interleaved import frozen,make_env
        from scripts.s1_resources import ResourceLog
        torch.set_num_threads(1 if args.split_process else 4)
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True)
        other=[p.strip() for p in pids.splitlines() if p.strip().isdigit() and int(p.strip())!=os.getpid()]
        if other:raise RuntimeError('GPU is not exclusive; active process IDs: '+','.join(other))
        if sha256(args.student)!=STUDENT_SHA:raise ValueError('Selected S1 identity mismatch')
        base=json.loads(args.baseline_manifest.read_text())
        manifest,suites=frozen.prepare_manifest(args,json.loads((frozen.ROOT/'configs/libero40_public_v1.json').read_text()))
        for field in ('checkpoint_files','versions','assets_sha256'):
            if manifest[field]!=base[field]:raise ValueError('Historical provenance mismatch: '+field)
        # Model/environment source identities must remain the evaluated version.
        for path,digest in base['sources'].items():
            if '/lerobot-' in path and (not Path(path).exists() or sha256(Path(path))!=digest):
                raise ValueError('Locked LeRobot source drift: '+path)
        tasks=[]
        for task in manifest['tasks']:
            if task['task_id'] not in (0,5):continue
            states=suites[task['suite']].get_task_init_states(task['task_id'])
            if len(states)<25:raise ValueError('Missing initial-state indices 20–24')
            task['state_sha256']=[state_hash(s) for s in states[:25]];tasks.append(task)
        import lerobot.policies.rtc.modeling_rtc as rtc_module
        sources=[Path(__file__),Path(__file__).with_name('s1_rtc_policy.py'),Path(__file__).with_name('s1_resources.py'),
            Path(__file__).parents[1]/'src/smolvla_flow/rtc_benchmark.py',Path(rtc_module.__file__),
            Path(run_libero_rollout.__file__),Path(__file__).with_name('train_distill40_stage.py'),
            Path(__file__).with_name('evaluate_s1_interleaved.py'),Path(__file__).with_name('report_s1_rtc.py')]
        if args.split_process:sources.append(Path(__file__).with_name('s1_rtc_process.py'))
        manifest.update(protocol=PROTOCOL,tasks=tasks,student_sha256=STUDENT_SHA,
            entry_sources={str(p):sha256(p) for p in sources},formal_schedule=schedule('formal'),check_schedule=schedule('check'))
        manifest['execution_runtime']=dict(version='split-process-v1' if args.split_process else 'thread-v1',
            controller_torch_threads=1 if args.split_process else 4,policy_torch_threads=4,batch=1)
        run_id=freeze_manifest(args.output/'manifest.json',manifest)
        if args.stage=='run' and list((args.output/'episodes/formal').rglob('*.json')):
            raise ValueError('Formal rows exist; use resume')
        if args.stage!='check':
            gate=json.loads((args.output/'gate.json').read_text())
            if not gate['passed'] or gate['run_id']!=run_id:raise ValueError('Missing or mismatched 24-episode gate')
        resources=ResourceLog(args.output)
        if args.split_process:
            from scripts.s1_rtc_process import ProcessPolicy
            policy=ProcessPolicy(args)
        else:policy=Policy(args)
        if args.stage=='check':
            task=tasks[0];env=make_env(suites[task['suite']],task)
            try:
                env.init_state_id=13;obs,_=env.reset(seed=123);relative_control_evidence(env)
                gate=policy.preflight(obs,task['language'])
            finally:env.close()
            atomic_json(args.output/'one_step_gate.json',dict(gate,run_id=run_id))
            if not gate['passed']:raise RuntimeError(gate['reason'])
        phase='check' if args.stage=='check' else 'formal'
        for job in schedule(phase):
            if args.execution_mode!='all' and job['mode']!=args.execution_mode:continue
            if args.delay_ms!='all' and job['delay_ms']!=int(args.delay_ms):continue
            task=next(t for t in tasks if (t['suite'],t['task_id'])==(job['suite'],job['task_id']))
            path=args.output/'episodes'/(job['key']+'.json')
            if path.exists():
                row=json.loads(path.read_text());validate_row(row,job,run_id)
                if row['init_state_sha256']!=task['state_sha256'][job['init_index']]:raise ValueError('Resume state drift')
                continue
            row=episode(args,job,task,suites[job['suite']],policy,run_id,resources)
            atomic_json(path,row);print('COMPLETE',job['key'],row['success'],flush=True)
        if phase=='check':
            for job in schedule('check'):
                validate_row(json.loads((args.output/'episodes'/(job['key']+'.json')).read_text()),job,run_id)
            resources.check();atomic_json(args.output/'gate.json',dict(passed=True,episodes=24,run_id=run_id))
        elif len(list((args.output/'episodes/formal').rglob('*.json')))==240:
            resources.check()
            from scripts.report_s1_rtc import report
            report(args.output)
    except BaseException:
        atomic_json(args.output/'error.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
    finally:
        if policy is not None and args.split_process:policy.close()
        if resources:resources.close()


if __name__=='__main__':
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    main()
