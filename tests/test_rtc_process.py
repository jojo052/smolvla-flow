import unittest
from scripts.s1_rtc_process import ProcessPolicy


def fake_worker(connection,args):
    connection.send((0,True,'ready'))
    while True:
        sequence,method,positional,keywords=connection.recv()
        if method=='close':break
        if method=='predict':
            connection.send((sequence,True,dict(observation=positional[0],metrics=dict(elapsed_s=0.,worker_call_s=0.))))
        elif method=='preflight':connection.send((sequence,False,'test worker error'))
        else:connection.send((sequence,True,None))
    connection.close()


class TestProcessPolicy(unittest.TestCase):
    def test_payload_order_reset_and_shutdown(self):
        policy=ProcessPolicy(None,worker=fake_worker)
        try:
            for i in range(3):
                result=policy.predict({'state':[i],'pixels':bytes(range(255))})
                self.assertEqual(result['observation']['state'],[i])
                self.assertEqual(result['observation']['pixels'],bytes(range(255)))
                self.assertGreater(result['metrics']['elapsed_s'],0)
            policy.reset()
            with self.assertRaisesRegex(RuntimeError,'test worker error'):policy.preflight()
        finally:policy.close()
        self.assertFalse(policy.process.is_alive())


if __name__=='__main__':unittest.main()
