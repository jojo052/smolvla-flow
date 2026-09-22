"""Recompute six-model results from immutable episode records, without reruns."""
import argparse
import json
from pathlib import Path
from smolvla_flow.benchmark40 import SUITES, atomic_json, fingerprint, validate_episode, summarize, latency


def load_rows(root, model):
    manifest=json.loads((root/'manifest.json').read_text())
    tasks={(t['suite'],t['task_id']):t for t in manifest['tasks']}
    expected={(s,t,i) for s in SUITES for t in range(10) for i in range(10)}
    rows=[];seen=set()
    for path in sorted((root/'formal').glob('*/*/task*/init*.json')):
        row=json.loads(path.read_text());key=(row['suite'],row['task_id'],row['init_index'])
        validate_episode(row,fingerprint(manifest),path.relative_to(root).with_suffix('').as_posix())
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unexpected episode')
        if row['init_state_sha256']!=tasks[key[:2]]['state_sha256'][key[2]]:
            raise ValueError('Initial-state identity mismatch')
        if model=='S1' and row['checkpoint_sha256']!=manifest['student_sha256']:
            raise ValueError('Student identity mismatch')
        seen.add(key);rows.append(dict(row,model=model,peak_vram_bytes=row.get('peak_vram_bytes',0)))
    if seen!=expected or (root/'error.json').exists():
        raise ValueError('Incomplete or errored evaluation: '+model)
    return manifest,rows


def parallel_metrics(root,rows):
    result=json.loads((root/'result.json').read_text())
    if not result['complete'] or {r['key']:r['success'] for r in result['episodes']}!={r['key']:r['success'] for r in rows}:
        raise ValueError('Result index does not match episodes')
    elapsed=result['end_to_end_seconds']
    telemetry=json.loads((root/'resources.json').read_text())
    if telemetry['error']:
        raise ValueError('Resource monitoring error')
    samples=telemetry['samples']
    return dict(end_to_end_seconds=elapsed,aggregate_tick_per_s=sum(r['steps'] for r in rows)/elapsed,
        success_per_hour=3600*sum(r['success'] for r in rows)/elapsed,
        server_prediction_latency=latency([x for r in rows for x in r['prediction_seconds']]),
        request_wait_latency=latency([x for r in rows for x in r['request_wait_seconds']]),
        mean_gpu_util=sum(x['gpu_util'] for x in samples)/len(samples),
        peak_total_gpu_mib=max(x['vram_mib'] for x in samples),
        includes_rendering_gpu_use=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('s1','t1','artifacts','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--s5',type=Path,help='Override historical S5 result location')
    args=p.parse_args()
    roots={m:args.artifacts/d for m,d in dict(T10='libero40_public_v1',T5='libero40_t5_direct_v1',
        S5='libero40_s5_b16_v1',T2='libero40_t2_direct_v1').items()}
    roots.update(T1=args.t1,S1=args.s1)
    if args.s5: roots['S5']=args.s5
    manifests={};records={}
    for model,root in roots.items():
        manifests[model],records[model]=load_rows(root,model)
    base=manifests['T1'];new=manifests['S1']
    for field in ('sources','versions','checkpoint_files','assets_sha256','protocol'):
        if base[field]!=new[field]:
            raise ValueError('T1/S1 comparison provenance drift: '+field)
    if new['execution']['batch']!=1 or new['execution']['workers']!=base['execution']['workers']:
        raise ValueError('Execution configuration differs')
    identity=lambda m:{(t['suite'],t['task_id']):(t['language'],t['bddl_sha256'],t['state_sha256'][:10]) for t in m['tasks']}
    # Entire task dictionaries can contain additional validation states; compare official test states only.
    for model,m in manifests.items():
        for field,value in base['protocol'].items():
            if field!='flow_steps' and m['protocol'][field]!=value:
                raise ValueError('Protocol differs for '+model)
        if identity(m)!=identity(base):
            raise ValueError('Task/test state differs for '+model)
    rows=[r for values in records.values() for r in values]
    result=summarize(rows,list(roots))
    for model in ('T1','S1'):
        for key in ('wall_tick_per_s','success_per_hour_including_reset','prediction_latency','select_latency','peak_vram_bytes'):
            result['models'][model].pop(key)
        result[model+'_parallel_efficiency']=parallel_metrics(roots[model],records[model])
    result['S1_minus_T1']=summarize(rows,['S1','T1'])['paired']
    result['S1_minus_T2']=summarize(rows,['S1','T2'])['paired']
    atomic_json(args.output/'results.json',result)
    lines=['# T2→S1 六模型对照','',
        '| 模型 | Spatial | Object | Goal | Long | 40任务宏平均 | 历史单环境预测 P50/P95 ms | 历史吞吐 tick/s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,r in result['models'].items():
        rates=[f"{100*r['suites'][s]['success_rate']:.2f}%" for s in SUITES]+[f"{100*r['macro_success_rate']:.2f}%"]
        timing=['并行指标单列','并行指标单列'] if model in ('T1','S1') else [
            f"{r['prediction_latency']['p50_ms']:.2f}/{r['prediction_latency']['p95_ms']:.2f}",f"{r['wall_tick_per_s']:.2f}"]
        lines.append('| '+' | '.join([model]+rates+timing)+' |')
    lines+=['','## 并行效率与配对比较','','```json',json.dumps({k:v for k,v in result.items()
        if k.endswith('_parallel_efficiency') or k.startswith('S1_minus_')},indent=2),'```','',
        'S1 教师目标采用批量计算，与历史单样本 T2 存在数值差异。使用已知测试初始状态，不作为未触及测试集的泛化证明。',
        '教师和历史模型没有追加正式评测。逐任务 Wilson 区间、失败类型和配对重采样区间见 results.json。']
    (args.output/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
