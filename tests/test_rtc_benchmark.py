import unittest
from smolvla_flow.rtc_benchmark import Timeline,schedule,wait_boundary,checked_chunk,PROTOCOL,validate_row,paired,accept_recorded
import json
import tempfile
from pathlib import Path
from scripts.report_s1_rtc import distribution,report
from smolvla_flow.benchmark40 import atomic_json,fingerprint

def chunk(value=1.): return [[value]*6+[-1.] for _ in range(50)]

class TestRTCProtocol(unittest.TestCase):
    def test_schedule(self):
        formal=schedule('formal');self.assertEqual(len(formal),240)
        self.assertEqual(len({x['key'] for x in formal}),240)
        self.assertEqual(formal,schedule('formal'));self.assertEqual(len(schedule('check')),24)
        self.assertEqual({x['init_index'] for x in formal},set(range(20,25)))
    def ready(self,mode):
        q=Timeline(mode);q.prime(chunk(),chunk(),.07)
        for i in range(10):q.pop(i)
        return q
    def test_sync_wait_is_zero_motion(self):
        q=self.ready('sync');q.submit(10)
        a,waiting,_=q.pop(10)
        self.assertEqual(a,(0.,0.,0.,0.,0.,0.,-1.));self.assertTrue(waiting)
    def test_async_preserves_tail_and_never_blends(self):
        q=self.ready('async');rid,prefix,d=q.submit(10)
        self.assertEqual(len(prefix),40);self.assertEqual(d,2)
        self.assertEqual(q.pop(10)[0],tuple(chunk()[0]))
        self.assertEqual(q.accept(rid,12,chunk(7),chunk(7)),2)
        self.assertEqual(len(q.queue),48);self.assertEqual(q.pop(12)[0][0],7.)
    def test_request_identity(self):
        q=self.ready('async');rid,_,_=q.submit(10)
        with self.assertRaises(ValueError):q.submit(11)
        with self.assertRaises(ValueError):q.accept(rid+1,12,chunk(),chunk())
    def test_rtc_horizon_and_expiry(self):
        for mode,delay in [('rtc',10),('async',50)]:
            q=self.ready(mode);rid,_,_=q.submit(10)
            with self.assertRaises(ValueError):q.accept(rid,10+delay,chunk(),chunk())
    def test_reset_and_nonfinite(self):
        q=Timeline('async');self.assertIsNone(q.pending);self.assertEqual(len(q.queue),0)
        with self.assertRaises(ValueError):checked_chunk(chunk(float('nan')))
    def test_absolute_clock(self):
        now=[10.];sleeps=[]
        def sleep(dt):sleeps.append(dt);now[0]+=dt
        for tick in range(5):wait_boundary(10.,tick,lambda:now[0],sleep);now[0]+=.012
        self.assertAlmostEqual(now[0],10.212)
        self.assertAlmostEqual(wait_boundary(10.,3,lambda:now[0],sleep),.062)
    def test_active_chunk_changes_only_on_acceptance(self):
        q=self.ready('async');rid,_,_=q.submit(10)
        self.assertEqual(q.active_sequence,0)
        q.pop(10);q.accept(rid,11,chunk(),chunk())
        self.assertEqual(q.active_sequence,rid)
    def test_rejected_response_keeps_timing_evidence(self):
        q=self.ready('rtc');rid,_,_=q.submit(10);records=[]
        result=dict(normalized=chunk(),actions=chunk(),metrics=dict(prediction_s=.35,elapsed_s=.48))
        with self.assertRaises(ValueError):accept_recorded(q,records,rid,20,result)
        self.assertEqual(records[0]['prediction_s'],.35)
        self.assertEqual(records[0]['actual_delay'],10)
        self.assertIsNone(records[0]['accepted_tick'])
        self.assertEqual(q.pending,(rid,10))
    def test_exhausted_async_holds_gripper(self):
        q=self.ready('async');q.submit(10)
        for tick in range(10,50):self.assertFalse(q.pop(tick)[1])
        self.assertEqual(q.pop(50),(tuple([0.]*6+[-1.]),True,None))
    def test_protocol_file_matches(self):
        path=Path(__file__).resolve().parents[1]/'configs/s1_rtc_realtime_v1.json'
        self.assertEqual(json.loads(path.read_text()),PROTOCOL)
    def test_resume_validation(self):
        job=schedule('formal')[0]
        row=dict(job,run_id='a',status='complete',success=True,steps=1,ticks=[{}],late_fraction=0,total_seconds=1)
        validate_row(row,job,'a')
        with self.assertRaises(ValueError):validate_row(row,job,'b')
        row['late_fraction']=.1
        with self.assertRaises(ValueError):validate_row(row,job,'a')
    def test_pairing_and_distribution(self):
        rows=[dict(j,success=j['mode']=='rtc',init_state_sha256=j['suite']+str(j['task_id'])+str(j['init_index'])) for j in schedule('formal')]
        result=paired(rows,0,'rtc','async')
        self.assertEqual(result['counts']['first_only'],40)
        self.assertEqual(result['task_bootstrap_95'],[1.,1.])
        self.assertEqual(distribution([0,1,2])['p50'],1.)
    def test_report_recompute_and_missing_episode_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            tasks=[dict(suite=s,task_id=t,state_sha256=['state']*25) for s in PROTOCOL['suites'] for t in (0,5)]
            manifest=dict(protocol=PROTOCOL,tasks=tasks,student_sha256='student')
            rid=fingerprint(manifest);atomic_json(root/'manifest.json',manifest)
            for name in ('gate','one_step_gate'):atomic_json(root/(name+'.json'),dict(passed=True,run_id=rid))
            for job in schedule('check')+schedule('formal'):
                request=dict(request_id=0,observed_tick=0,accepted_tick=0,initial=True,
                    actual_delay=0,estimated_delay=0,prediction_s=.05,elapsed_s=.05,
                    rtc_processor_inclusive_s=0,rtc_extra_s=0,denoise_forward_s=.04,
                    injected_wait_s=0,forward_calls=1,rtc_calls=0)
                tick=dict(waiting=False,lateness_s=0,observation_age_ticks=0,chunk_boundary_jump=None)
                row=dict(job,run_id=rid,status='complete',success=True,steps=1,ticks=[tick],requests=[request],
                    late_fraction=0,total_seconds=1,control_seconds=.05,init_state_sha256='state',student_sha256='student')
                atomic_json(root/'episodes'/(job['key']+'.json'),row)
            result=report(root);self.assertEqual(result['episodes'],240)
            self.assertTrue(all(g['success']==40 for g in result['groups']))
            self.assertEqual(result,report(root))
            path=root/'episodes'/(schedule('formal')[0]['key']+'.json')
            path.unlink()
            with self.assertRaises(ValueError):report(root)

if __name__=='__main__':unittest.main()
