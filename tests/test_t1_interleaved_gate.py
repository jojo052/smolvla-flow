"""CPU-only tests of the exact gate function, without importing GPU adapters."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np

source = Path(__file__).resolve().parents[1]/'scripts/run_t1_interleaved.py'
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'compare_pilots')
namespace = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
compare = namespace['compare_pilots']


class GateTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(suite=s,init_state_sha256=s,steps=2,success=True,
                         noise_sha256=['n0','n1'],normalized_actions=[[.2]*7,[.4]*7])
                     for s in ('a','b','c','d')]
    def test_same(self):
        self.assertTrue(compare(self.rows,deepcopy(self.rows))['passed'])
    def test_completion_order(self):
        self.assertTrue(compare(self.rows,list(reversed(deepcopy(self.rows))))['passed'])
    def test_action_drift(self):
        other=deepcopy(self.rows); other[0]['normalized_actions'][0][0]+=.001
        self.assertFalse(compare(self.rows,other)['passed'])
    def test_seed_drift(self):
        other=deepcopy(self.rows); other[0]['noise_sha256'][0]='wrong'
        self.assertFalse(compare(self.rows,other)['passed'])
    def test_outcome_drift(self):
        other=deepcopy(self.rows); other[0]['success']=False
        self.assertFalse(compare(self.rows,other)['passed'])
    def test_length_drift(self):
        other=deepcopy(self.rows); other[0]['normalized_actions'].pop(); other[0]['steps']=1
        self.assertFalse(compare(self.rows,other)['passed'])
    def test_missing_suite(self):
        self.assertFalse(compare(self.rows,self.rows[:3])['passed'])
    def test_duplicate_suite(self):
        self.assertFalse(compare(self.rows,[self.rows[0]]*4)['passed'])


if __name__=='__main__': unittest.main()
