"""Recompute the completed paired benchmark from immutable local episode records."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from smolvla_flow.benchmark40 import fingerprint, validate_episode, summarize, markdown_report, atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows, manifests, completions = [], [], {}
    for name in ['libero40_public_v1', 'libero40_act_v1']:
        folder = ROOT / 'artifacts' / name
        manifest = json.loads((folder / 'manifest.json').read_text())
        manifests.append(manifest)
        status = json.loads((folder / 'status.json').read_text())
        assert status['status'] == 'complete' and status['formal_complete'] == 400
        completions[name] = datetime.fromtimestamp(status['updated_at'], ZoneInfo('Asia/Shanghai')).isoformat()
        selected = []
        for path in sorted((folder / 'formal').glob('*/*/task*/init*.json')):
            row = json.loads(path.read_text())
            validate_episode(row, fingerprint(manifest), str(path.relative_to(folder).with_suffix('')))
            task = next(t for t in manifest['tasks'] if t['suite'] == row['suite'] and t['task_id'] == row['task_id'])
            assert row['init_state_sha256'] == task['state_sha256'][row['init_index']]
            selected.append(row)
        assert len(selected) == len({r['key'] for r in selected}) == 400
        for task in manifest['tasks']:
            subset = [r for r in selected if r['suite'] == task['suite'] and r['task_id'] == task['task_id']]
            assert sorted(r['init_index'] for r in subset) == list(range(10))
            for outcome in [True, False]:
                if any(r['success'] == outcome for r in subset):
                    video = folder / 'formal' / subset[0]['model'] / task['suite'] / ('task%02d' % task['task_id']) / ('first_success.mp4' if outcome else 'first_failure.mp4')
                    assert video.is_file() and video.stat().st_size > 0
        rows.extend(selected)
    a, b = manifests
    assert a['tasks'] == b['tasks']
    assert a['assets_sha256'] == b['assets_sha256']
    assert a['versions'] == b['versions']
    shared_sources = set(a['sources']) & set(b['sources'])
    assert all(a['sources'][s] == b['sources'][s] for s in shared_sources)
    for key in ['suites','initial_state_indices','settle_steps','policy_seed','environment_seed','mode','rtc','observation_size','action_processing']:
        assert a['protocol'][key] == b['protocol'][key]
    result = summarize(rows, ['smolvla_official', 'act_public'])
    result['completed_at'] = completions
    result['verification'] = {'unique_formal_episodes': 800, 'same_tasks_states_assets_versions_shared_sources': True,
                              'required_videos_present': True, 'pilot_excluded': True}
    out = ROOT / 'artifacts/libero40_comparison_v1'
    atomic_json(out / 'results.json', result)
    report = markdown_report(result)
    report += '\n## 完成时间\n\n' + '\n'.join(f'- {k}: {v}' for k,v in completions.items()) + '\n'
    report += '\n## 配对比较\n\n```json\n' + json.dumps(result['paired'], indent=2) + '\n```\n'
    report += '\n## 效率\n\n| 模型 | 混合 select_action 均值 ms | 分配显存峰值 GiB | 成功回合/小时（含失败与重置） |\n|---|---:|---:|---:|\n'
    for model, data in result['models'].items():
        report += f"| {model} | {data['select_latency']['mean_ms']:.3f} | {data['peak_vram_bytes']/2**30:.3f} | {data['success_per_hour_including_reset']:.2f} |\n"
    report += '\n## 解释边界\n\nACT 为 jamongsteak/act_libero 无语言条件公开权重，每次预测执行100动作；SmolVLA 为官方10 Flow steps、每步重新预测。训练数据版本与规模未控制，结果仅适用于这些checkpoint和原生执行配置，不能证明ACT架构普遍较弱。单任务专门训练的历史ACT结果不并入本轮。低ACT成功率的具体成因需要独立诊断，严格加载与有限动作检查不等同于全部语义正确性证明。\n\n双方各400正式回合、初始状态0至9、任务与资产及共享环境代码一致。视频已回传，报告未对视频内容进行完整人工审阅。权重哈希在远端无卡实例再次核对通过。\n'
    (out / 'results.md').write_text(report)
    print(json.dumps({'completed_at': completions, 'paired': result['paired'], 'verification': result['verification']}, indent=2))


if __name__ == '__main__':
    main()
