import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.report_distill40 import main
from smolvla_flow.benchmark40 import SUITES, fingerprint


class ReportTests(unittest.TestCase):
    def test_default_still_requires_four_groups(self):
        with patch('sys.argv', ['report', '--t10', '/tmp/t10', '--s5', '/tmp/s5', '--output', '/tmp/unused-report']), self.assertRaises(SystemExit):
            main()

    def test_s5_comparison_has_all_800_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = {k: 'fixed' for k in ('suites','settle_steps','policy_seed','environment_seed','mode','rtc','observation_size','action_processing')}
            tasks = [dict(suite=s,task_id=t,name=f'{s}/{t}',language='task',bddl_sha256='b',
                          init_file_sha256='i',max_steps=limit,state_sha256=[str(i) for i in range(10)])
                     for s,limit in SUITES.items() for t in range(10)]
            for model in ('T10','S5'):
                folder = root/model
                folder.mkdir()
                manifest = {'protocol':protocol,'tasks':tasks,'model':model}
                (folder/'manifest.json').write_text(json.dumps(manifest))
                for task in tasks:
                    destination = folder/'formal'/model/task['suite']/f'task{task["task_id"]:02d}'
                    destination.mkdir(parents=True)
                    for i in range(10):
                        row = dict(model=model,suite=task['suite'],task_id=task['task_id'],init_index=i,
                            key=f'formal/{model}/{task["suite"]}/task{task["task_id"]:02d}/init{i:02d}',run_id=fingerprint(manifest),status='complete',
                            phase='formal',init_state_sha256=str(i),action_nonfinite_count=0,steps=1,
                            prediction_seconds=[.1],select_seconds=[.1],success=True,failure_type=None,
                            elapsed_seconds=1.,total_seconds=2.,peak_vram_bytes=100)
                        (destination/f'init{i:02d}.json').write_text(json.dumps(row))
            with patch('sys.argv',['report','--s5-only','--t10',str(root/'T10'),'--s5',str(root/'S5'),'--output',str(root/'report')]):
                main()
            result=json.loads((root/'report/results.json').read_text())
            self.assertEqual(set(result['models']),{'T10','S5'})
            self.assertEqual(result['paired_comparisons']['S5-T10']['counts']['both_success'],400)
            self.assertTrue(all(r['complete'] for r in result['models'].values()))
            damaged = next((root/'S5/formal').glob('*/*/task*/init*.json'))
            row = json.loads(damaged.read_text())
            row['prediction_seconds'] = [float('nan')]
            damaged.write_text(json.dumps(row))
            with patch('sys.argv',['report','--s5-only','--t10',str(root/'T10'),'--s5',str(root/'S5'),'--output',str(root/'invalid')]), self.assertRaises(ValueError):
                main()
            self.assertFalse((root/'invalid/results.json').exists())


if __name__ == '__main__':
    unittest.main()
