import unittest
from smolvla_flow.distill40_selection import SUITES, CANDIDATES, select_candidate, validate_selection_episodes

def candidate(update,successes,loss):
    rows=[]
    for suite in SUITES:
        for task in range(10):
            for index in (11,12):
                rows.append(dict(suite=suite,task_id=task,init_index=index,status='complete',
                                 success=len(rows)<successes,checkpoint_sha256=str(update),
                                 policy_seed=123,environment_seed=123))
    return dict(update=update,episodes=rows,checkpoint_sha256=str(update),offline_loss=loss)

class SelectionTests(unittest.TestCase):
    def test_fixed_ranking(self):
        cs=[candidate(n,40,1.) for n in CANDIDATES]
        self.assertEqual(select_candidate(cs)['update'],5000)
        cs[1]['offline_loss']=.5
        self.assertEqual(select_candidate(cs)['update'],10000)
        cs[2]=candidate(15000,41,9.)
        self.assertEqual(select_candidate(cs)['update'],15000)

    def test_missing_candidate(self):
        with self.assertRaises(ValueError): select_candidate([candidate(5000,40,1.)])

    def test_formal_indices_rejected(self):
        c=candidate(5000,40,1.);c['episodes'][0]['init_index']=0
        with self.assertRaises(ValueError): validate_selection_episodes(c['episodes'],'5000')

    def test_duplicate_rejected(self):
        c=candidate(5000,40,1.);c['episodes'][-1]=c['episodes'][0]
        with self.assertRaises(ValueError): validate_selection_episodes(c['episodes'],'5000')

if __name__=='__main__': unittest.main()
