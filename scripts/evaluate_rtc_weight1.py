"""Independent 80-episode RTC weight=1 follow-up; historical controls are read-only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback

from smolvla_flow.rtc_benchmark import PROTOCOL, schedule, validate_row, paired
from smolvla_flow.benchmark40 import atomic_json, sha256, freeze_manifest, fingerprint


def jobs():
    return [j for j in schedule('formal') if j['mode']=='rtc']


def report(args):
    from scripts.report_s1_rtc import distribution
    manifest=json.loads((args.output/'manifest.json').read_text());run_id=fingerprint(manifest)
    if manifest['protocol']!=dict(PROTOCOL,guidance_weight=1.):raise ValueError('Protocol drift')
    if (args.output/'error.json').exists():raise ValueError('Unresolved error')
    expected={args.output/'episodes'/(j['key']+'.json') for j in jobs()}
    if set((args.output/'episodes/formal').rglob('*.json'))!=expected:raise ValueError('Expected exactly 80 RTC episodes')
    rows=[]
    for j in jobs():
        row=json.loads((args.output/'episodes'/(j['key']+'.json')).read_text())
        validate_row(row,j,run_id)
        task=next(t for t in manifest['tasks'] if t['suite']==j['suite'] and t['task_id']==j['task_id'])
        if row['init_state_sha256']!=task['state_sha256'][j['init_index']]:raise ValueError('State drift')
        if row['student_sha256']!=manifest['student_sha256']:raise ValueError('Student drift')
        if any(q.get('guidance_weight_cap')!=1. for q in row['requests']):raise ValueError('Guidance drift')
        if row.get('video') and not (args.output/row['video']).exists():raise ValueError('Missing video')
        rows.append(row)
    original=[]
    for rel,digest in manifest['historical_files'].items():
        path=args.source/rel
        if sha256(path)!=digest:raise ValueError('Historical file changed: '+rel)
        if rel.startswith('episodes/formal/'):
            original.append(json.loads(path.read_text()))
    old_manifest=json.loads((args.source/'manifest.json').read_text());old_id=fingerprint(old_manifest)
    by_key={r['key']:r for r in original}
    for job in schedule('formal'):validate_row(by_key[job['key']],job,old_id)
    groups=[]
    for delay in (0,100):
        rr=[r for r in rows if r['delay_ms']==delay];ticks=[t for r in rr for t in r['ticks']]
        requests=[q for r in rr for q in r['requests']];seconds=sum(r['total_seconds'] for r in rr)
        jumps=[t['chunk_boundary_jump'] for t in ticks if t['chunk_boundary_jump']]
        groups.append(dict(mode='rtc',guidance=1.,delay_ms=delay,success=sum(r['success'] for r in rr),episodes=40,
            success_rate=sum(r['success'] for r in rr)/40,total_seconds=seconds,
            success_per_hour=sum(r['success'] for r in rr)*3600/seconds,
            prediction_seconds=distribution(q['prediction_s'] for q in requests),
            queue_empty_fraction=sum(t['waiting'] for t in ticks)/len(ticks),
            late_fraction=sum(t['lateness_s']>.005 for t in ticks)/len(ticks),
            jumps={k:distribution(x[k] for x in jumps) for k in ('translation','rotation','gripper')}))
    combined=rows+[r for r in original if r['mode']!='rtc']
    comparisons=[dict(delay_ms=d,**paired(combined,d,'rtc','async')) for d in (0,100)]
    atomic_json(args.output/'report.json',dict(complete=True,episodes=80,run_id=run_id,groups=groups,
        paired_comparisons=comparisons,limitations=['Historical controls reused; runs were not concurrent.',
        'Follow-up after seeing weight10 outcomes; known initial states, not untouched test generalization.']))


def main():
    p=argparse.ArgumentParser()
    for key in ('output','checkpoint','student','assets-dir','source','check-output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=('run','resume','report'),default='run')
    args=p.parse_args();args.guidance=1.
    if args.stage=='report':report(args);return
    if args.stage=='run' and args.output.exists():raise ValueError('New independent output required')
    if (args.output/'error.json').exists():raise ValueError('Inspect error before any authorized recovery')
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=json.loads((args.source/'manifest.json').read_text())
    check=json.loads((args.check_output/'summary.json').read_text())
    if not check['complete'] or check['episodes']!=16 or check['guidance']!=1.:raise ValueError('Low weight check missing')
    if source['protocol']!=PROTOCOL:raise ValueError('Original protocol drift')
    for path,digest in source['entry_sources'].items():
        if sha256(Path(path))!=digest:raise ValueError('Source drift: '+path)
    for rel,digest in source['checkpoint_files'].items():
        if sha256(args.checkpoint/rel)!=digest:raise ValueError('Checkpoint drift: '+rel)
    from scripts.evaluate_s1_rtc import STUDENT_SHA, episode, state_hash
    if sha256(args.student)!=STUDENT_SHA:raise ValueError('S1 identity')
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(x.strip().isdigit() for x in pids.splitlines()):raise ValueError('GPU is occupied')
    history={str(f.relative_to(args.source)):sha256(f) for f in (args.source/'episodes/formal').rglob('*.json')}
    if len(history)!=240:raise ValueError('Expected 240 historical records')
    history['manifest.json']=sha256(args.source/'manifest.json')
    run_id=freeze_manifest(args.output/'manifest.json',dict(protocol=dict(PROTOCOL,guidance_weight=1.),
        tasks=source['tasks'],student_sha256=STUDENT_SHA,historical_files=history,jobs=jobs(),
        low_weight_check_sha256=sha256(args.check_output/'summary.json'),
        entry_sources={str(f):sha256(f) for f in [Path(__file__),Path(__file__).with_name('check_rtc_low_guidance.py')]},
        original_run_id=fingerprint(source),execution_runtime=source['execution_runtime'],
        limitation='RTC-only follow-up, historical controls reused; initial states already examined'))
    import torch
    from scripts import run_libero_rollout as loader
    from scripts.evaluate_s1_interleaved import make_env
    from scripts.s1_rtc_process import ProcessPolicy
    from scripts.check_rtc_low_guidance import worker
    from scripts.s1_resources import ResourceLog
    torch.set_num_threads(1);loader._configure_libero(args.assets_dir)
    import libero.libero as core
    from libero.libero import benchmark
    core._assets_path_cache=str(args.assets_dir)
    suites={s:benchmark.get_benchmark_dict()[s]() for s in PROTOCOL['suites']}
    policy=None;resources=ResourceLog(args.output)
    try:
        policy=ProcessPolicy(args,worker=worker)
        task=source['tasks'][0];env=make_env(suites[task['suite']],task)
        try:
            env.init_state_id=13;obs,_=env.reset(seed=123)
            for _ in range(2):policy.predict(obs,task['language'],'async',0)
        finally:env.close()
        for job in jobs():
            task=next(t for t in source['tasks'] if t['suite']==job['suite'] and t['task_id']==job['task_id'])
            if state_hash(suites[job['suite']].get_task_init_states(job['task_id'])[job['init_index']])!=task['state_sha256'][job['init_index']]:
                raise ValueError('Initial state drift')
            target=args.output/'episodes'/(job['key']+'.json')
            if target.exists():
                validate_row(json.loads(target.read_text()),job,run_id);continue
            row=episode(args,job,task,suites[job['suite']],policy,run_id,resources)
            atomic_json(target,row);print('COMPLETE',job['key'],row['success'],flush=True)
        resources.check();report(args)
    except BaseException:
        atomic_json(args.output/'error.json',dict(traceback=traceback.format_exc()));raise
    finally:
        if policy is not None:policy.close()
        resources.close()


if __name__=='__main__':
    os.environ.setdefault('MUJOCO_GL','egl')
    main()
