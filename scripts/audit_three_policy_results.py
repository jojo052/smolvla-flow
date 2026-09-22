"""CPU-only provenance audit and episode pairing; never runs a policy."""
import argparse
import json
from pathlib import Path

from smolvla_flow.benchmark40 import fingerprint, validate_episode, atomic_json, sha256


def audit(roots, output):
    manifests = {}; records = {}; checks = []; files = {}
    expected = {(s, t, i) for s in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10') for t in range(10) for i in range(10)}
    for model, root in roots.items():
        manifest = json.loads((root / 'manifest.json').read_text())
        manifests[model] = manifest
        tasks = {(t['suite'], t['task_id']): t for t in manifest['tasks']}
        assert len(tasks) == len(manifest['tasks']) == 40
        rows = {}
        for path in sorted((root / 'formal').glob('*/*/task*/init*.json')):
            r = json.loads(path.read_text()); key = (r['suite'], r['task_id'], r['init_index'])
            assert key in expected and key not in rows, (model, key)
            validate_episode(r, fingerprint(manifest), path.relative_to(root).with_suffix('').as_posix())
            assert r['phase'] == 'formal'
            task = tasks[key[:2]]
            assert r['task_name'] == task['name'] and r['language'] == task['language']
            assert r['action_processing'] == manifest['protocol']['action_processing']
            expected_hash = manifest.get('distillation_evaluation', {}).get('student_sha256', manifest['checkpoint_files']['model.safetensors'])
            if model != 'T10' or 'checkpoint_sha256' in r:
                assert r['checkpoint_sha256'] == expected_hash
            assert r['init_state_sha256'] == tasks[key[:2]]['state_sha256'][key[2]]
            for field in ('policy_seed', 'environment_seed', 'settle_steps'):
                assert r[field] == manifest['protocol'][field]
            assert len(r['prediction_seconds']) == r['steps']
            rows[key] = r; files[str(path)] = sha256(path)
        assert set(rows) == expected, (model, len(rows))
        records[model] = rows; files[str(root / 'manifest.json')] = sha256(root / 'manifest.json')
        missing_hash = sum('checkpoint_sha256' not in r for r in rows.values())
        if missing_hash:
            checks.append(f'{model}: {missing_hash} episodes have no separate checkpoint hash; provenance relies on run_id binding to manifest')
        checks.append(f'{model}: 400/400 unique complete episodes; provenance, states, seeds, budgets, finite actions and per-step predictions PASS')
    base = manifests['T10']
    differences = {}
    for model in ('T5', 'S5'):
        m = manifests[model]; differences[model] = {}
        for field in ('checkpoint_files', 'sources', 'versions', 'python', 'gpu', 'gpu_total_memory', 'gpu_driver', 'assets_sha256'):
            same = base[field] == m[field]
            checks.append(f'T10/{model} {field}: {"MATCH" if same else "DIFFERENT"}')
            if not same: differences[model][field] = {'T10': base[field], model: m[field]}
        for field in set(base['protocol']) | set(m['protocol']):
            if base['protocol'].get(field) != m['protocol'].get(field):
                differences[model]['protocol.' + field] = [base['protocol'].get(field), m['protocol'].get(field)]
                assert field == 'flow_steps', field
        a = {(t['suite'], t['task_id']): t for t in base['tasks']}
        for t in m['tasks']:
            other = a[t['suite'], t['task_id']]
            for field in ('name', 'language', 'bddl_sha256', 'init_file_sha256', 'max_steps'):
                assert t[field] == other[field], (model, field)
            assert t['state_sha256'][:10] == other['state_sha256'][:10]
        checks.append(f'T10/{model}: all 40 task identities and 400 initial states MATCH')
        assert all(records[model][k]['input_contract'] == records['T10'][k]['input_contract'] for k in expected)
        checks.append(f'T10/{model}: all recorded image/state input contracts MATCH')
    pairs = []
    for key in sorted(expected):
        p = dict(suite=key[0], task_id=key[1], init_index=key[2], task_name=records['T10'][key]['task_name'] if 'task_name' in records['T10'][key] else records['T10'][key].get('task'))
        for model in roots:
            r = records[model][key]
            p[model] = {k: r.get(k) for k in ('success', 'steps', 'failure_type', 'init_state_sha256')}
        p['S5_T5_category'] = ('both_success' if p['S5']['success'] and p['T5']['success'] else 'S5_only' if p['S5']['success'] else 'T5_only' if p['T5']['success'] else 'both_failure')
        pairs.append(p)
    summary = {}; task_rows = []
    for suite in base['protocol']['suites']:
        selected = [p for p in pairs if p['suite'] == suite]
        summary[suite] = {m: sum(p[m]['success'] for p in selected) for m in roots}
        summary[suite]['pairs'] = {c: sum(p['S5_T5_category'] == c for p in selected) for c in ('both_success', 'S5_only', 'T5_only', 'both_failure')}
        summary[suite]['T5_only_T10_success'] = sum(p['S5_T5_category'] == 'T5_only' and p['T10']['success'] for p in selected)
        for task in range(10):
            ps = [p for p in selected if p['task_id'] == task]
            name = next(t['name'] for t in base['tasks'] if t['suite'] == suite and t['task_id'] == task)
            task_rows.append(dict(suite=suite, task_id=task, name=name, **{m: sum(p[m]['success'] for p in ps) for m in roots}, T5_only=[p['init_index'] for p in ps if p['S5_T5_category'] == 'T5_only'], S5_only=[p['init_index'] for p in ps if p['S5_T5_category'] == 'S5_only']))
    atomic_json(output / 'audit.json', dict(checks=checks, differences=differences, summary=summary, source_files_sha256=files, manifests=manifests))
    atomic_json(output / 'paired_episodes.json', pairs)
    atomic_json(output / 'paired_tasks.json', task_rows)
    lines = ['# T10 / T5 / S5 无卡审计', '', '仅检查已有记录，不运行推理。清单一致不等于原始噪声、观测张量已验证一致。', '', '## 完整性与协议', ''] + ['- ' + c for c in checks]
    lines += ['', '## 套件配对', '', '|Suite|T10|T5|S5|双方成功|仅T5成功|仅S5成功|双方失败|仅T5成功中T10也成功|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s, v in summary.items():
        c = v['pairs']; lines.append(f"|{s}|{v['T10']}|{v['T5']}|{v['S5']}|{c['both_success']}|{c['T5_only']}|{c['S5_only']}|{c['both_failure']}|{v['T5_only_T10_success']}|")
    lines += ['', '## 40任务表（每模型每任务10回合）', '', '|Suite|Task|T10|T5|S5|仅T5成功init|仅S5成功init|任务名|', '|---|---:|---:|---:|---:|---|---|---|']
    for t in task_rows:
        lines.append(f"|{t['suite']}|{t['task_id']}|{t['T10']}|{t['T5']}|{t['S5']}|{t['T5_only']}|{t['S5_only']}|{t['name']}|")
    lines += ['', '## 400回合配对表', '', '1=成功，0=超时失败。', '', '|Suite|Task|Init|T10|T5|S5|S5/T5配对|', '|---|---:|---:|---:|---:|---:|---|']
    for p in pairs:
        lines.append(f"|{p['suite']}|{p['task_id']}|{p['init_index']}|{int(p['T10']['success'])}|{int(p['T5']['success'])}|{int(p['S5']['success'])}|{p['S5_T5_category']}|")
    lines += ['', '## 限制', '', '- 权重与处理器一致性依据运行清单；本报告不重新加载模型。', '- 仅检查清单覆盖的代码及依赖，不保证未记录的全部运行时状态一致。', '- 未保存的观测、动作与噪声张量无法从成功率记录还原。', '- S5额外加载学生权重；T5仅变更采样步数。入口包装代码需与清单哈希及加载逻辑另行核验。', '- 失败标签仅为timeout，不代表已定位抓取或释放等物理失败原因。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for m in ('t10', 't5', 's5'):
        parser.add_argument('--' + m, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit({m: getattr(args, m.lower()) for m in ('T10', 'T5', 'S5')}, args.output)
