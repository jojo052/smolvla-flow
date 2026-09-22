"""Pure bookkeeping for the independent real-time S1 experiment."""
import math
import random
from collections import deque

SUITES = {'libero_spatial':220, 'libero_object':280, 'libero_goal':300, 'libero_10':520}
MODES = ('sync', 'async', 'rtc')
PROTOCOL = dict(version='s1-rtc-realtime-v1', suites=SUITES, tasks=[0,5],
    initial_states=[20,21,22,23,24], check_initial_state=13, seed=123,
    control_hz=20, settle_ticks=10, chunk_size=50, flow_steps=1, execute_steps=10,
    modes=list(MODES), delays_ms=[0,100], guidance_weight=10., schedule='EXP',
    execution_horizon=10, max_lateness_s=.005, max_late_fraction=.05,
    waiting_action='zero_motion_last_environment_gripper', batch=1, environments=1)


def schedule(phase):
    if phase not in ('check','formal'): raise ValueError('Unknown phase')
    pairs=[(s,t,i) for s in SUITES for t in ([0] if phase=='check' else [0,5])
           for i in ([13] if phase=='check' else range(20,25))]
    random.Random(123).shuffle(pairs)
    conditions=[(m,d) for d in (0,100) for m in MODES]
    result=[]
    for j,(s,t,i) in enumerate(pairs):
        ordered=conditions[j%6:]+conditions[:j%6]
        for m,d in ordered:
            result.append(dict(phase=phase,suite=s,task_id=t,init_index=i,mode=m,delay_ms=d,
                key=f'{phase}/{s}/task{t:02d}/init{i:02d}/{m}_d{d}'))
    return result


def checked_chunk(chunk):
    if len(chunk)!=50 or any(len(a)!=7 for a in chunk): raise ValueError('Chunk must be 50x7')
    if any(not math.isfinite(float(x)) for a in chunk for x in a): raise ValueError('Nonfinite chunk')
    return [tuple(map(float,a)) for a in chunk]


class Timeline:
    """No interpolation. Only the control thread mutates queue and request state."""
    def __init__(self,mode):
        if mode not in MODES: raise ValueError('Unknown mode')
        self.mode=mode;self.queue=deque();self.pending=None;self.executed=0
        self.sequence=0;self.active_sequence=0;self.origin=0;self.last_gripper=None;self.last_delay=0
    def prime(self,normalized,actions,elapsed):
        self.queue=deque(zip(checked_chunk(normalized),checked_chunk(actions)))
        self.last_delay=math.ceil(elapsed*20)
        if self.mode=='rtc' and self.last_delay>=10: raise ValueError('Initial RTC delay exceeds horizon')
    def should_request(self):
        return self.pending is None and self.executed>=10
    def submit(self,tick):
        if not self.should_request(): raise ValueError('Unexpected or duplicate request')
        self.sequence+=1;self.pending=(self.sequence,tick)
        prefix=[a for a,_ in self.queue]
        if self.mode=='sync': self.queue.clear()
        return self.sequence,prefix,self.last_delay
    def accept(self,request_id,tick,normalized,actions):
        if self.pending is None or self.pending[0]!=request_id: raise ValueError('Response identity mismatch')
        normalized=checked_chunk(normalized);actions=checked_chunk(actions)
        delay=tick-self.pending[1]
        if delay<0 or delay>=50: raise ValueError('Expired or future response')
        if self.mode=='rtc' and delay>=10: raise ValueError('RTC delay exceeds horizon')
        self.active_sequence=request_id
        self.origin=self.pending[1];self.pending=None;self.last_delay=delay;self.executed=0
        self.queue=deque(zip(normalized[delay:],actions[delay:]))
        return delay
    def pop(self,tick):
        if self.queue:
            _,action=self.queue.popleft();self.executed+=1;self.last_gripper=action[6]
            return action,False,tick-self.origin
        if self.last_gripper is None: raise ValueError('Unprimed queue')
        return (0.,0.,0.,0.,0.,0.,self.last_gripper),True,None


def wait_boundary(start,tick,clock,sleep):
    deadline=start+tick/20
    remaining=deadline-clock()
    if remaining>0: sleep(remaining)
    return max(0.,clock()-deadline)


def accept_recorded(timeline,requests,request_id,tick,result):
    """Record a completed request before a rejecting admission gate can raise."""
    observed=timeline.pending[1] if timeline.pending is not None else None
    entry=dict(request_id=request_id,accepted_tick=None,attempted_tick=tick,
        observed_tick=observed,actual_delay=None if observed is None else tick-observed,
        initial=False,**result['metrics'])
    requests.append(entry)
    actual=timeline.accept(request_id,tick,result['normalized'],result['actions'])
    entry['accepted_tick']=tick
    return actual


def validate_row(row,job,run_id):
    if row.get('run_id')!=run_id or any(row.get(k)!=v for k,v in job.items()):
        raise ValueError('Episode identity/configuration mismatch')
    if row.get('status')!='complete' or type(row.get('success')) is not bool:
        raise ValueError('Incomplete episode')
    if not 1<=row['steps']<=SUITES[job['suite']]: raise ValueError('Invalid step count')
    if not row['success'] and row['steps']!=SUITES[job['suite']]: raise ValueError('Premature failure')
    if len(row['ticks'])!=row['steps']: raise ValueError('Missing tick metrics')
    if row['late_fraction']>.05: raise ValueError('Control deadline gate failed')
    if not math.isfinite(row['total_seconds']) or row['total_seconds']<=0: raise ValueError('Invalid duration')


def paired(rows,delay,first,second):
    maps=[{(r['suite'],r['task_id'],r['init_index']):r for r in rows
           if r['mode']==m and r['delay_ms']==delay} for m in (first,second)]
    if len(maps[0])!=40 or maps[0].keys()!=maps[1].keys(): raise ValueError('Incomplete pairing')
    counts=dict(both_success=0,first_only=0,second_only=0,both_fail=0);deltas=[]
    for s in SUITES:
        for t in (0,5):
            differences=[]
            for i in range(20,25):
                a,b=(m[s,t,i] for m in maps)
                if a['init_state_sha256']!=b['init_state_sha256']: raise ValueError('Unpaired states')
                x,y=a['success'],b['success']
                counts['both_success' if x and y else 'first_only' if x else 'second_only' if y else 'both_fail']+=1
                differences.append(int(x)-int(y))
            deltas.append(sum(differences)/5)
    rng=random.Random(123);boot=sorted(sum(rng.choices(deltas,k=8))/8 for _ in range(10000))
    return dict(first=first,second=second,counts=counts,difference=sum(deltas)/8,
                task_bootstrap_95=[boot[249],boot[9749]],tasks=8)
