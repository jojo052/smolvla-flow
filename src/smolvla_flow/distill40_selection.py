"""Fixed closed-loop checkpoint selection, independent of formal results."""
import math

SUITES = ('libero_spatial','libero_object','libero_goal','libero_10')
CANDIDATES = (5000,10000,15000,20000)


def validate_selection_episodes(rows, checkpoint_sha256):
    expected={(suite,task,index) for suite in SUITES for task in range(10) for index in (11,12)}
    seen=set()
    for row in rows:
        key=(row['suite'],row['task_id'],row['init_index'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate selection episode')
        seen.add(key)
        if row['status']!='complete' or type(row['success']) is not bool:
            raise ValueError('Incomplete selection episode')
        if row['checkpoint_sha256']!=checkpoint_sha256:
            raise ValueError('Selection checkpoint mismatch')
        if row['policy_seed']!=123 or row['environment_seed']!=123:
            raise ValueError('Selection seed mismatch')
    if seen!=expected:
        raise ValueError('All eighty selection episodes required')
    return sum(row['success'] for row in rows)


def select_candidate(candidates, expected_updates=CANDIDATES):
    if len(candidates)!=4 or {c['update'] for c in candidates}!=set(expected_updates):
        raise ValueError('All four scheduled candidates required')
    ranked=[]
    for c in candidates:
        score=validate_selection_episodes(c['episodes'],c['checkpoint_sha256'])
        loss=c['offline_loss']
        if not math.isfinite(loss) or loss<0:
            raise ValueError('Invalid offline loss')
        ranked.append((-score,loss,c['update'],c))
    return min(ranked,key=lambda x:x[:3])[3]
