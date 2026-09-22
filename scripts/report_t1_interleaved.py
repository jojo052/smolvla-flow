"""Recompute T1 results, keeping parallel efficiency out of historical columns."""
import argparse
import json
from pathlib import Path
from smolvla_flow.benchmark40 import SUITES, atomic_json, fingerprint, validate_episode, summarize, latency


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1',type=Path,required=True,help='formal_run directory')
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    roots={'T10':'libero40_public_v1','T5':'libero40_t5_direct_v1',
           'S5':'libero40_s5_b16_v1','T2':'libero40_t2_direct_v1'}
    all_rows=[]; manifests={}
    expected={(s,t,i) for s in SUITES for t in range(10) for i in range(10)}
    for model,directory in dict(roots,T1=None).items():
        root=args.t1 if model=='T1' else args.artifacts/directory
        m=json.loads((root/'manifest.json').read_text()); manifests[model]=m
        seen=set(); tasks={(t['suite'],t['task_id']):t for t in m['tasks']}
        for path in sorted((root/'formal').glob('*/*/task*/init*.json')):
            row=json.loads(path.read_text()); k=(row['suite'],row['task_id'],row['init_index'])
            validate_episode(row,fingerprint(m),path.relative_to(root).with_suffix('').as_posix())
            assert k not in seen and k in expected
            seen.add(k)
            assert row['init_state_sha256']==tasks[k[:2]]['state_sha256'][k[2]]
            all_rows.append(dict(row,model=model,peak_vram_bytes=row.get('peak_vram_bytes',0)))
        assert seen==expected, (model,len(seen))
    base=manifests['T10']; new=manifests['T1']
    for field in ('sources','versions','checkpoint_files','assets_sha256'):
        assert base[field]==new[field],field
    for field in base['protocol']:
        if field!='flow_steps': assert base['protocol'][field]==new['protocol'][field],field
    assert new['protocol']['flow_steps']==1 and new['execution']['batch']==1
    final=json.loads((args.t1/'result.json').read_text())
    assert final['complete'] and len(final['episodes'])==400 and not (args.t1/'error.json').exists()
    records=[r for r in all_rows if r['model']=='T1']
    assert {r['key']:r['success'] for r in final['episodes']}=={r['key']:r['success'] for r in records}
    result=summarize(all_rows,list(roots)+['T1'])
    result['T1_minus_T2']=summarize(all_rows,['T1','T2'])['paired']
    # These serial aggregations are not valid wall-clock efficiency for T1.
    for key in ('wall_tick_per_s','success_per_hour_including_reset','prediction_latency','select_latency','peak_vram_bytes'):
        result['models']['T1'].pop(key)
    telemetry=json.loads((args.t1/'resources.json').read_text())
    samples=telemetry['samples']; elapsed=final['end_to_end_seconds']
    result['T1_parallel_efficiency']={
        'scope':'interleaved GPU batch=1; model load and serial pilots excluded; worker initialization, resets and video writing included',
        'end_to_end_seconds':elapsed,'aggregate_tick_per_s':sum(r['steps'] for r in records)/elapsed,
        'success_per_hour':3600*sum(r['success'] for r in records)/elapsed,
        'batch_size':1,'server_prediction_latency':latency([x for r in records for x in r['prediction_seconds']]),
        'request_wait_latency':latency([x for r in records for x in r['request_wait_seconds']]),
        'mean_gpu_util':sum(x['gpu_util'] for x in samples)/len(samples),
        'gpu_below_20_fraction':sum(x['gpu_util']<20 for x in samples)/len(samples),
        'peak_total_gpu_mib':max(x['vram_mib'] for x in samples),
        'peak_container_memory_bytes':max(x['container_memory_bytes'] for x in samples),
        'includes_rendering_gpu_use':True}
    atomic_json(args.output/'results.json',result)
    lines=['# T1 与既有四组对照','',
           '| 模型 | Spatial | Object | Goal | Long | 宏平均 | 历史单环境预测 P50/P95 ms | 历史吞吐 tick/s |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,r in result['models'].items():
        rates=[f"{100*r['suites'][s]['success_rate']:.2f}%" for s in SUITES]+[f"{100*r['macro_success_rate']:.2f}%"]
        if model=='T1': timing=['单列并行指标','单列并行指标']
        else: timing=[f"{r['prediction_latency']['p50_ms']:.2f}/{r['prediction_latency']['p95_ms']:.2f}",f"{r['wall_tick_per_s']:.2f}"]
        lines.append('| '+ ' | '.join([model]+rates+timing)+' |')
    lines += ['', '## T1 并行效率及配对结果', '', '```json',
              json.dumps({k:result[k] for k in ('T1_parallel_efficiency','T1_minus_T2')},indent=2), '```',
              '', '逐任务成功次数、Wilson 区间及超时数见 results.json 的 tasks。旧模型未重跑。']
    (args.output/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
