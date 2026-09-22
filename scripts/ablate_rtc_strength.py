"""Fixed 96-episode strength-only ablation, independent from prior evaluations."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

from smolvla_flow.rtc_benchmark import PROTOCOL, validate_row
from smolvla_flow.benchmark40 import atomic_json, sha256, freeze_manifest, fingerprint


def schedule():
    pairs=[(s,t,i) for s in PROTOCOL['suites'] for t in (0,5) for i in (25,26)]
    random.Random(123).shuffle(pairs)
    conditions=[(w,d) for d in (0,100) for w in (0.,.5,1.)]
    jobs=[]
    for n,(s,t,i) in enumerate(pairs):
        for w,d in conditions[n%6:]+conditions[:n%6]:
            phase='ablation_w'+str(w).replace('.','p');mode='async' if w==0 else 'rtc'
            jobs.append(dict(phase=phase,suite=s,task_id=t,init_index=i,mode=mode,delay_ms=d,guidance=w,
                key=f'{phase}/{s}/task{t:02d}/init{i:02d}/{mode}_d{d}'))
    return jobs


def worker(connection,args):
    try:
        import torch
        import resource
        from scripts.s1_rtc_policy import Policy
        torch.set_num_threads(4);policy=Policy(args);weight=None
        connection.send((0,True,'ready'))
        while True:
            seq,method,pos,kw=connection.recv()
            if method=='close':break
            start=time.perf_counter()
            if method=='set_guidance':
                weight=pos[0]
                if weight not in (0.,.5,1.):raise ValueError('Unapproved strength')
                policy.rtc.rtc_config.max_guidance_weight=weight;result=weight
            elif method in ('reset','predict'):
                if weight is None:raise ValueError('Missing strength')
                result=getattr(policy,method)(*pos,**kw)
                if method=='predict':
                    if (pos[2]=='rtc')!=(weight>0):raise ValueError('Mode/strength mismatch')
                    result['metrics'].update(guidance_weight_cap=weight,worker_call_s=time.perf_counter()-start,
                        worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            else:raise ValueError('Unknown RPC')
            connection.send((seq,True,result))
    except BaseException:
        connection.send((locals().get('seq',0),False,traceback.format_exc()))
    finally:connection.close()


def report(output):
    from scripts.report_s1_rtc import distribution
    manifest=json.loads((output/'manifest.json').read_text());run_id=fingerprint(manifest)
    if (output/'error.json').exists():raise ValueError('Unresolved error')
    expected={output/'episodes'/(j['key']+'.json') for j in schedule()}
    if set((output/'episodes').rglob('*.json'))!=expected:raise ValueError('Expected 96 unique episodes')
    rows=[]
    for job in schedule():
        r=json.loads((output/'episodes'/(job['key']+'.json')).read_text());validate_row(r,job,run_id)
        task=next(t for t in manifest['tasks'] if (t['suite'],t['task_id'])==(r['suite'],r['task_id']))
        if r['init_state_sha256']!=task['state_sha256'][r['init_index']] or r['student_sha256']!=manifest['student_sha256']:
            raise ValueError('Identity drift')
        if any(q['guidance_weight_cap']!=job['guidance'] for q in r['requests']):raise ValueError('Strength drift')
        if r.get('video') and not (output/r['video']).exists():raise ValueError('Missing video')
        rows.append(r)
    groups=[];comparisons=[]
    for d in (0,100):
        for w in (0.,.5,1.):
            rr=[r for r in rows if r['guidance']==w and r['delay_ms']==d]
            ticks=[t for r in rr for t in r['ticks']];requests=[q for r in rr for q in r['requests']]
            jumps=[t['chunk_boundary_jump'] for t in ticks if t['chunk_boundary_jump']]
            sec=sum(r['total_seconds'] for r in rr);wins=sum(r['success'] for r in rr)
            groups.append(dict(guidance=w,delay_ms=d,success=wins,episodes=16,success_rate=wins/16,
                total_seconds=sec,success_per_hour=3600*wins/sec,
                prediction_seconds=distribution(q['prediction_s'] for q in requests),
                queue_empty_fraction=sum(t['waiting'] for t in ticks)/len(ticks),
                late_fraction=sum(t['lateness_s']>.005 for t in ticks)/len(ticks),
                jumps={k:distribution(x[k] for x in jumps) for k in ('translation','rotation','gripper')}))
        for a,b in ((.5,0.),(1.,0.),(.5,1.)):
            maps=[{(r['suite'],r['task_id'],r['init_index']):r for r in rows if r['guidance']==w and r['delay_ms']==d} for w in (a,b)]
            counts=dict(both_success=0,first_only=0,second_only=0,both_fail=0);delta=[]
            for s in PROTOCOL['suites']:
                for t in (0,5):
                    values=[]
                    for i in (25,26):
                        x,y=(m[s,t,i]['success'] for m in maps)
                        counts['both_success' if x and y else 'first_only' if x else 'second_only' if y else 'both_fail']+=1
                        values.append(int(x)-int(y))
                    delta.append(sum(values)/2)
            rng=random.Random(123);boot=sorted(sum(rng.choices(delta,k=8))/8 for _ in range(10000))
            comparisons.append(dict(delay_ms=d,first=a,second=b,counts=counts,difference=sum(delta)/8,
                task_bootstrap_95=[boot[249],boot[9749]]))
    atomic_json(output/'report.json',dict(complete=True,episodes=96,groups=groups,comparisons=comparisons,
        limitations=['Eight tasks, two states each; exploratory intervals.',
        'New saved-run initial-state indices, not new task generalization. No additional strength search.']))


def main():
    p=argparse.ArgumentParser()
    for k in ('output','checkpoint','student','assets-dir','source'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--stage',choices=('run','report'),default='run');args=p.parse_args()
    if args.stage=='report':report(args.output);return
    if args.output.exists():raise ValueError('Use a new ablation directory')
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(x.strip().isdigit() for x in pids.splitlines()):raise ValueError('GPU occupied')
    source=json.loads((args.source/'manifest.json').read_text())
    if source['protocol']!=PROTOCOL:raise ValueError('Original protocol drift')
    for path,want in source['entry_sources'].items():
        if sha256(Path(path))!=want:raise ValueError('Code drift: '+path)
    for path,want in source['checkpoint_files'].items():
        if sha256(args.checkpoint/path)!=want:raise ValueError('Weight/processor drift: '+path)
    from scripts.evaluate_s1_rtc import STUDENT_SHA,episode,state_hash
    if sha256(args.student)!=STUDENT_SHA:raise ValueError('S1 drift')
    # Search saved episode/error records, including other directory conventions.
    hits=[];scanned=0
    for path in args.source.parent.rglob('*.json'):
        if path.name in ('manifest.json','report.json','summary.json'):continue
        try:r=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if isinstance(r,dict) and ('steps' in r or 'ticks' in r):
            scanned+=1
            if r.get('init_index',r.get('initial_state_index')) in (25,26):hits.append(str(path))
    if hits:raise ValueError('Initial states already used: '+str(hits))
    import torch
    from scripts import run_libero_rollout as loader
    from scripts.evaluate_s1_interleaved import make_env
    from scripts.s1_rtc_process import ProcessPolicy
    from scripts.s1_resources import ResourceLog
    torch.set_num_threads(1);loader._configure_libero(args.assets_dir)
    import libero.libero as core
    from libero.libero import benchmark
    core._assets_path_cache=str(args.assets_dir)
    suites={s:benchmark.get_benchmark_dict()[s]() for s in PROTOCOL['suites']}
    tasks=source['tasks']
    for task in tasks:
        states=suites[task['suite']].get_task_init_states(task['task_id'])
        if len(states)<27:raise ValueError('Missing init25/26')
        hashes=[state_hash(s) for s in states[:27]]
        if hashes[:25]!=task['state_sha256']:raise ValueError('Historical states drift')
        task['state_sha256']=hashes
    args.output.mkdir(parents=True)
    run_id=freeze_manifest(args.output/'manifest.json',dict(protocol=dict(PROTOCOL,initial_states=[25,26],guidance_weight=None,
        guidance_weights=[0.,.5,1.],modes=['async','rtc']),
        tasks=tasks,jobs=schedule(),student_sha256=STUDENT_SHA,source_manifest_sha256=sha256(args.source/'manifest.json'),
        checkpoint_files=source['checkpoint_files'],execution_runtime=source['execution_runtime'],
        entry_sha256=sha256(Path(__file__)),usage_scan=dict(saved_record_count=scanned,matches=hits),
        limitation='No saved prior use found; does not establish teacher pretraining non-overlap.'))
    resources=ResourceLog(args.output);policy=None
    try:
        policy=ProcessPolicy(args,worker=worker);policy.call('set_guidance',0.)
        task=tasks[0];env=make_env(suites[task['suite']],task)
        try:
            env.init_state_id=13;obs,_=env.reset(seed=123)
            for _ in range(2):policy.predict(obs,task['language'],'async',0)
        finally:env.close()
        for job in schedule():
            policy.call('set_guidance',job['guidance'])
            task=next(t for t in tasks if (t['suite'],t['task_id'])==(job['suite'],job['task_id']))
            row=episode(args,job,task,suites[job['suite']],policy,run_id,resources)
            atomic_json(args.output/'episodes'/(job['key']+'.json'),row)
            print('COMPLETE',job['key'],row['success'],flush=True)
        resources.check();report(args.output)
    except BaseException:
        atomic_json(args.output/'error.json',dict(traceback=traceback.format_exc()));raise
    finally:
        if policy is not None:policy.close()
        resources.close()


if __name__=='__main__':
    os.environ.setdefault('MUJOCO_GL','egl')
    main()
