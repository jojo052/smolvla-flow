import copy
import unittest
from unittest.mock import patch
from smolvla_flow.distill40_prefetch import ObservationPrefetch

class Sampler:
    def __init__(self): self.i=0
    def sample(self):
        self.i+=1
        return (0,0,self.i)

class Reader:
    def read(self,*key): return key
    def stats(self): return {}

class TestPrefetch(unittest.TestCase):
    @patch('smolvla_flow.distill40_prefetch.decode_observation',side_effect=lambda x:x)
    def test_order_and_pending_recovery(self,_):
        sampler=Sampler();pre=ObservationPrefetch(sampler,Reader())
        try:
            self.assertEqual(sampler.i,0)
            for _ in range(3):
                rows,keys,_=pre.next();self.assertEqual(rows,keys)
            saved=copy.deepcopy(sampler)
            expected=pre.next()
        finally: pre.close()
        restored=ObservationPrefetch(saved,Reader())
        try: self.assertEqual(expected,restored.next())
        finally: restored.close()

    @patch('smolvla_flow.distill40_prefetch.decode_observation',side_effect=ValueError('bad image'))
    def test_error_does_not_advance_sampler(self,_):
        sampler=Sampler();pre=ObservationPrefetch(sampler,Reader())
        try:
            with self.assertRaises(ValueError): pre.next()
            self.assertEqual(sampler.i,0)
        finally: pre.close()
