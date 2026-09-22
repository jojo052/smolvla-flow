"""Single policy subprocess RPC. Environment/rendering remain in the controller process."""
import multiprocessing as mp
import threading
import time
import traceback


def policy_worker(connection,args):
    try:
        import torch
        import resource
        from scripts import run_libero_rollout
        from scripts.s1_rtc_policy import Policy
        torch.set_num_threads(4)
        policy=Policy(args)
        connection.send((0,True,'ready'))
        while True:
            sequence,method,positional,keywords=connection.recv()
            if method=='close':break
            if method not in ('predict','reset','preflight'):raise ValueError('Unknown method')
            start=time.perf_counter()
            result=getattr(policy,method)(*positional,**keywords)
            if method=='predict':
                result['metrics']['worker_call_s']=time.perf_counter()-start
                result['metrics']['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            connection.send((sequence,True,result))
    except BaseException:
        try:connection.send((locals().get('sequence',0),False,traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:connection.close()


class ProcessPolicy:
    def __init__(self,args,worker=policy_worker):
        context=mp.get_context('spawn')
        self.connection,other=context.Pipe()
        self.process=context.Process(target=worker,args=(other,args))
        self.lock=threading.Lock();self.sequence=0
        self.process.start();other.close()
        try:self.receive(0,180)
        except BaseException:self.close();raise

    def receive(self,sequence,timeout):
        if not self.connection.poll(timeout):raise TimeoutError('Policy worker response timeout')
        received,ok,value=self.connection.recv()
        if received!=sequence:raise RuntimeError('Policy RPC response identity mismatch')
        if not ok:raise RuntimeError('Policy worker failed:\n'+value)
        return value

    def call(self,method,*args,**kwargs):
        with self.lock:
            self.sequence+=1;sequence=self.sequence;start=time.perf_counter()
            self.connection.send((sequence,method,args,kwargs))
            result=self.receive(sequence,180 if method=='preflight' else 30)
            elapsed=time.perf_counter()-start
            if method=='predict':
                metrics=result['metrics']
                metrics['native_request_s']=metrics['elapsed_s']
                metrics['elapsed_s']=elapsed
                metrics['ipc_and_dispatch_s']=max(0.,elapsed-metrics['worker_call_s'])
            return result

    def predict(self,*args,**kwargs):return self.call('predict',*args,**kwargs)
    def reset(self):return self.call('reset')
    def preflight(self,*args,**kwargs):return self.call('preflight',*args,**kwargs)
    def close(self):
        if self.process.is_alive():
            try:self.connection.send((self.sequence+1,'close',(),{}))
            except (BrokenPipeError,OSError):pass
            self.process.join(5)
        if self.process.is_alive():self.process.terminate();self.process.join()
        self.connection.close()
