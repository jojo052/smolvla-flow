"""Conditional small closed-loop diagnostic; no writes to original benchmark."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback


def worker(connection,args):
    try:
        import torch
        import resource
        from scripts.s1_rtc_policy import Policy
        torch.set_num_threads(4)
        policy=Policy(args)
        policy.rtc.rtc_config.max_guidance_weight=args.guidance
        connection.send((0,True,'ready'))
        while True:
            sequence,method,positional,keywords=connection.recv()
            if method=='close': break
            if method not in ('predict','reset'):raise ValueError('Unexpected RPC')
            start=time.perf_counter()
            result=getattr(policy,method)(*positional,**keywords)
            if method=='predict':
                result['metrics']['worker_call_s']=time.perf_counter()-start
                result['metrics']['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                result['metrics']['guidance_weight_cap']=args.guidance
            connection.send((sequence,True,result))
    except BaseException:
        connection.send((locals().get('sequence',0),False,traceback.format_exc()))
    finally: connection.close()


def main():
    p=argparse.ArgumentParser()
    for key in ('output','diagnosis','checkpoint','student','assets-dir','source'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new diagnostic directory')
    from smolvla_flow.benchmark40 import atomic_json,sha256,freeze_manifest
    from scripts.evaluate_s1_rtc import episode,STUDENT_SHA,state_hash
    from scripts.s1_rtc_process import ProcessPolicy
    from scripts.s1_resources import ResourceLog
    diagnosis=json.loads((args.diagnosis/'summary.json').read_text())
    args.guidance=diagnosis['selected_weight']
    if args.guidance is None or not 0<args.guidance<=1:raise ValueError('No eligible guidance candidate')
    cases=[r for r in diagnosis['rows'] if r['model']=='S1' and r['weight']==args.guidance]
    if not cases or not all(r['gate'] for r in cases):raise ValueError('Diagnostic gate failed')
    if sha256(args.student)!=STUDENT_SHA:raise ValueError('S1 identity mismatch')
    source=json.loads((args.source/'manifest.json').read_text())
    for path,digest in source['entry_sources'].items():
        if sha256(Path(path))!=digest:raise ValueError('Source drift: '+path)
    args.output.mkdir(parents=True)
    import torch
    from scripts import run_libero_rollout as loader
    from scripts.evaluate_s1_interleaved import make_env
    torch.set_num_threads(1)
    loader._configure_libero(args.assets_dir)
    import libero.libero as core
    from libero.libero import benchmark
    core._assets_path_cache=str(args.assets_dir)
    tasks=[t for t in source['tasks'] if t['task_id']==0]
    suites={t['suite']:benchmark.get_benchmark_dict()[t['suite']]() for t in tasks}
    jobs=[dict(phase='check',suite=t['suite'],task_id=0,init_index=13,mode=m,delay_ms=d,
          key=f"check/{t['suite']}/task00/init13/{m}_d{d}") for t in tasks for d in (0,100) for m in ('async','rtc')]
    random.Random(123).shuffle(jobs)
    run_id=freeze_manifest(args.output/'manifest.json',dict(source_manifest_sha256=sha256(args.source/'manifest.json'),
        diagnosis_sha256=sha256(args.diagnosis/'summary.json'),guidance=args.guidance,student_sha256=STUDENT_SHA,
        entry_sha256=sha256(Path(__file__)),jobs=jobs,protocol=dict(source['protocol'],guidance_weight=args.guidance),
        purpose='16 diagnostic episodes on known check states; not formal evaluation'))
    policy=None;resources=ResourceLog(args.output);rows=[]
    try:
        policy=ProcessPolicy(args,worker=worker)
        t=tasks[0];env=make_env(suites[t['suite']],t)
        try:
            env.init_state_id=13;obs,_=env.reset(seed=123)
            for _ in range(2):policy.predict(obs,t['language'],'async',0)
        finally:env.close()
        for job in jobs:
            task=next(t for t in tasks if t['suite']==job['suite'])
            if state_hash(suites[job['suite']].get_task_init_states(0)[13])!=task['state_sha256'][13]:
                raise ValueError('Initial state drift')
            row=episode(args,job,task,suites[job['suite']],policy,run_id,resources)
            atomic_json(args.output/'episodes'/(job['key']+'.json'),row);rows.append(row)
            print('COMPLETE',job['key'],row['success'],flush=True)
        atomic_json(args.output/'summary.json',dict(complete=True,episodes=len(rows),guidance=args.guidance,
            groups=[dict(mode=m,delay_ms=d,success=sum(r['success'] for r in rows if r['mode']==m and r['delay_ms']==d),
                episodes=4) for d in (0,100) for m in ('async','rtc')]))
    except BaseException:
        atomic_json(args.output/'error.json',dict(traceback=traceback.format_exc()));raise
    finally:
        if policy is not None:policy.close()
        resources.close()


if __name__=='__main__':
    os.environ.setdefault('MUJOCO_GL','egl')
    main()
