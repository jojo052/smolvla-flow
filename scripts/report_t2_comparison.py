"""Validate completed T2 and recompute comparison with existing checkpoints."""
import json
from pathlib import Path
from smolvla_flow.benchmark40 import SUITES, fingerprint, validate_episode, summarize, markdown_report, atomic_json

roots = dict(T10='libero40_public_v1', T5='libero40_t5_direct_v1', S5='libero40_s5_b16_v1', T2='libero40_t2_direct_v1')
rows = []; manifests = {}
expected = {(s,t,i) for s in SUITES for t in range(10) for i in range(10)}
for model, folder in roots.items():
    root = Path('artifacts') / folder
    m = json.loads((root / 'manifest.json').read_text()); manifests[model] = m
    tasks = {(t['suite'],t['task_id']):t for t in m['tasks']}
    assert len(tasks) == len(m['tasks']) == 40
    seen = set(); selected = []
    for p in sorted((root / 'formal').glob('*/*/task*/init*.json')):
        r = json.loads(p.read_text()); k = (r['suite'],r['task_id'],r['init_index'])
        validate_episode(r, fingerprint(m), p.relative_to(root).with_suffix('').as_posix())
        assert k in expected and k not in seen
        seen.add(k)
        assert r['init_state_sha256'] == tasks[k[:2]]['state_sha256'][k[2]]
        assert len(r['prediction_seconds']) == r['steps']
        for f in ('policy_seed','environment_seed','settle_steps','action_processing'):
            assert r[f] == m['protocol'][f]
        assert r['task_name'] == tasks[k[:2]]['name']
        if model != 'T10':
            assert r['checkpoint_sha256'] == m['distillation_evaluation']['student_sha256']
        selected.append(r); rows.append(dict(r,model=model))
    assert seen == expected
    if model == 'T2':
        final = json.loads((root/'result.json').read_text())
        assert final['complete'] and final['successes'] == sum(r['success'] for r in selected)
        assert sorted(final['episodes'],key=lambda r:r['key']) == sorted(selected,key=lambda r:r['key'])
        assert not (root/'error.json').exists()
    if model != 'T10':
        base = manifests['T10']
        for field in ('sources','versions','checkpoint_files','assets_sha256'):
            assert base[field] == m[field], (model,field)
        for field in base['protocol']:
            if field != 'flow_steps': assert base['protocol'][field] == m['protocol'][field]
        for t in m['tasks']:
            other = next(x for x in base['tasks'] if (x['suite'],x['task_id']) == (t['suite'],t['task_id']))
            for field in ('name','language','bddl_sha256','init_file_sha256','max_steps'):
                assert other[field] == t[field]
            assert other['state_sha256'][:10] == t['state_sha256'][:10]
assert manifests['T2']['protocol']['flow_steps'] == 2
result = summarize(rows,list(roots))
result['paired_comparisons'] = {f'T2-{m}':summarize(rows,['T2',m])['paired'] for m in ('T10','T5','S5')}
result['source_manifest_fingerprints'] = {m:fingerprint(v) for m,v in manifests.items()}
out = Path('artifacts/libero40_t2_comparison_v1')
atomic_json(out/'results.json',result)
(out/'results.md').write_text(markdown_report(result)+'\n\nAll four groups have 400 validated episodes. T2 final result matches episode files. Prior runs reused; timing excludes video encoding. Runtime noise tensors and actual integrator-call counts were not audited here.\n\n```json\n'+json.dumps(result['paired_comparisons'],indent=2)+'\n```\n')
print(json.dumps({'models':result['models'],'pairs':result['paired_comparisons']},indent=2))
