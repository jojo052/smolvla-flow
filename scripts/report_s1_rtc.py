"""Recompute the independent real-time experiment; reject incomplete results."""
import json
import math
from pathlib import Path
from smolvla_flow.benchmark40 import atomic_json,fingerprint
from smolvla_flow.rtc_benchmark import PROTOCOL,MODES,schedule,validate_row,paired


def distribution(values):
    values=sorted(float(v) for v in values if v is not None)
    if not values:return dict(n=0,mean=None,p50=None,p95=None,max=None)
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite report metric')
    def quantile(q):
        index=(len(values)-1)*q;lo=int(index);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return dict(n=len(values),mean=sum(values)/len(values),p50=quantile(.5),p95=quantile(.95),max=max(values))


def report(output):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text());run_id=fingerprint(manifest)
    if manifest['protocol']!=PROTOCOL:raise ValueError('Protocol mismatch')
    if (output/'error.json').exists():raise ValueError('Unresolved execution error')
    for name in ('gate.json','one_step_gate.json'):
        gate=json.loads((output/name).read_text())
        if not gate['passed'] or gate['run_id']!=run_id:raise ValueError('Missing preflight gate')
    jobs=schedule('formal');expected={output/'episodes'/(j['key']+'.json') for j in jobs}
    if set((output/'episodes/formal').rglob('*.json'))!=expected:raise ValueError('Expected exactly 240 formal records')
    rows=[]
    for job in schedule('check')+jobs:
        row=json.loads((output/'episodes'/(job['key']+'.json')).read_text());validate_row(row,job,run_id)
        task=next(t for t in manifest['tasks'] if (t['suite'],t['task_id'])==(job['suite'],job['task_id']))
        if row['init_state_sha256']!=task['state_sha256'][job['init_index']]:raise ValueError('State identity mismatch')
        if row['student_sha256']!=manifest['student_sha256']:raise ValueError('Student identity mismatch')
        if row.get('video') and not (output/row['video']).is_file():raise ValueError('Missing recorded video')
        if job['phase']=='formal':rows.append(row)
    groups=[]
    for delay in (0,100):
        for mode in MODES:
            selected=[r for r in rows if r['mode']==mode and r['delay_ms']==delay]
            ticks=[t for r in selected for t in r['ticks']]
            requests=[q for r in selected for q in r['requests']]
            intervals=[]
            for row in selected:
                starts=[q['observed_tick'] for q in row['requests']]
                intervals.extend(b-a for a,b in zip(starts,starts[1:]))
            seconds=sum(r['total_seconds'] for r in selected);success=sum(r['success'] for r in selected)
            groups.append(dict(mode=mode,delay_ms=delay,episodes=len(selected),success=success,
                success_rate=success/40,total_episode_seconds=seconds,success_per_hour=success*3600/seconds,
                episode_seconds=distribution(r['total_seconds'] for r in selected),
                control_seconds=sum(r['control_seconds'] for r in selected),
                queue_empty_fraction=sum(t['waiting'] for t in ticks)/len(ticks),
                deadline_violation_fraction=sum(t['lateness_s']>.005 for t in ticks)/len(ticks),
                prediction_seconds=distribution(q['prediction_s'] for q in requests),
                request_seconds=distribution(q['elapsed_s'] for q in requests),
                ipc_and_dispatch_seconds=distribution(q.get('ipc_and_dispatch_s') for q in requests),
                policy_peak_rss_bytes=distribution(q.get('worker_peak_rss_bytes') for q in requests),
                rtc_processor_inclusive_seconds=distribution(q['rtc_processor_inclusive_s'] for q in requests),
                rtc_extra_seconds=distribution(q['rtc_extra_s'] for q in requests),
                denoise_forward_seconds=distribution(q['denoise_forward_s'] for q in requests),
                injected_wait_seconds=distribution(q['injected_wait_s'] for q in requests),
                observation_age_ticks=distribution(t['observation_age_ticks'] for t in ticks),
                actual_delay_ticks=distribution(q['actual_delay'] for q in requests if not q['initial']),
                estimated_delay_ticks=distribution(q['estimated_delay'] for q in requests if not q['initial']),
                replan_interval_ticks=distribution(intervals),
                forward_calls=sum(q['forward_calls'] for q in requests),rtc_calls=sum(q['rtc_calls'] for q in requests),
                chunk_jumps={axis:distribution(t['chunk_boundary_jump'][axis] for t in ticks if t['chunk_boundary_jump'])
                             for axis in ('translation','rotation','gripper')}))
    contrasts=[dict(delay_ms=d,**paired(rows,d,a,b)) for d in (0,100) for a,b in [('rtc','async'),('async','sync')]]
    resource_path=output/'resources.jsonl'
    resource_rows=[json.loads(line) for line in resource_path.read_text().splitlines()] if resource_path.exists() else []
    result=dict(run_id=run_id,complete=True,episodes=240,groups=groups,paired_comparisons=contrasts,
        resource_scope='All recorded check/run/resume processes, including initialization and checks',
        resources={key:distribution(r[key] for r in resource_rows) for key in
                   ('gpu_utilization','gpu_used_mib','process_peak_rss_bytes','container_memory_bytes')},
        limitations=['Eight task clusters; confidence intervals are exploratory.',
            'Known benchmark task family with new initial-state indices 20-24.',
            'Different execution protocol from historical six-model evaluations.',
            'RTC processor timing includes its native forward and input-gradient work; not additive to prediction timing.',
            'Prediction distributions include initial and outstanding final requests.',
            'Episode wall time includes reset, first prediction, control, pending request drain and video encoding; model load is separate.'])
    atomic_json(output/'report.json',result)
    text=['# S1 real-time chunk benchmark','',f'Run: `{run_id}`','']
    for delay in (0,100):
        text += [f'## Added delay: {delay} ms','',
            '| Mode | Success/40 | Rate | Mean episode s | Success/hour | Queue empty | Prediction P50/P95 ms |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for group in groups:
            if group['delay_ms']!=delay:continue
            p=group['prediction_seconds']
            text.append(f"| {group['mode']} | {group['success']}/40 | {group['success_rate']:.2%} | {group['episode_seconds']['mean']:.2f} | {group['success_per_hour']:.2f} | {group['queue_empty_fraction']:.2%} | {p['p50']*1000:.2f}/{p['p95']*1000:.2f} |")
        text.append('')
    text+=['## Limitations','']+['- '+x for x in result['limitations']]
    # Generated report, atomically replaced like the JSON source of truth.
    temporary=output/'report.md.tmp';temporary.write_text('\n'.join(text)+'\n');temporary.replace(output/'report.md')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    report(p.parse_args().output)
