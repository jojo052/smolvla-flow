"""One-shot checkpoint-gated isolated probe and conservative training restart.

This supervisor is deliberately scoped to the approved S5 batch16 run.
It never changes optimizer settings and never launches S2 or powers off.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def identity(pid):
    root=Path('/proc')/str(pid)
    return (root.joinpath('stat').read_text().split(') ',1)[1].split()[19],
            root.joinpath('cmdline').read_bytes())

def main():
    p=argparse.ArgumentParser()
    for name in ('trainer','pipeline','queue'): p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args()
    root=Path('/root/autodl-tmp/outputs/distill40_s5_b16_v1')
    audit=root/'prefetch_probe';audit.mkdir(exist_ok=True)
    ids={pid:identity(pid) for pid in (a.trainer,a.pipeline,a.queue)}
    for pid,marker in ((a.trainer,b'train_distill40_stage.py'),(a.pipeline,b'run_distill40_stage_pipeline.py'),(a.queue,b'queue_distill40_s5_eval.py')):
        if marker not in ids[pid][1] or str(root).encode() not in ids[pid][1]:
            raise RuntimeError('Process scope mismatch')
    argv=lambda pid:[v.decode() for v in ids[pid][1].split(b'\0') if v]
    pipeline_argv=argv(a.pipeline);queue_argv=argv(a.queue)
    env=dict(v.split('=',1) for v in (Path('/proc')/str(a.trainer)/'environ').read_bytes().decode().split('\0') if '=' in v)
    cwd=(Path('/proc')/str(a.trainer)/'cwd').resolve()
    def record(event,**fields):
        row=dict(event=event,time=time.time(),**fields)
        with (audit/'switch.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def send(pid,sig):
        if identity(pid)!=ids[pid]:raise RuntimeError('Process identity changed')
        os.kill(pid,sig)
    record('waiting_for_checkpoint',update=1000,identities={str(k):v[0] for k,v in ids.items()})
    ckpt=root/'stage1/001000.pt';validation=root/'stage1/offline_001000.json'
    while True:
        for pid in ids:
            if identity(pid)!=ids[pid]:raise RuntimeError('Process exited before checkpoint gate')
        if ckpt.exists() and validation.exists():
            try:r=json.loads(validation.read_text())
            except json.JSONDecodeError:time.sleep(15);continue
            pairs=r['pairs'];counts=collections.Counter(x['task'] for x in pairs)
            if len(pairs)!=1280 or len(counts)!=40 or set(counts.values())!={32}:raise RuntimeError('Invalid validation coverage')
            if not all(math.isfinite(x[k]) for x in pairs for k in ('loss','trajectory','endpoint')):raise RuntimeError('Nonfinite validation')
            h=hashlib.sha256()
            with ckpt.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            if h.hexdigest()!=r['checkpoint_sha256']:raise RuntimeError('Checkpoint hash mismatch')
            break
        time.sleep(15)  # Server-local wait; no model/API polling.
    # Hold parent and old evaluation watcher before terminating their trainer.
    send(a.queue,signal.SIGSTOP);send(a.pipeline,signal.SIGSTOP)
    send(a.trainer,signal.SIGTERM)
    for _ in range(30):
        status=Path('/proc')/str(a.trainer)/'status'
        if not status.exists() or '\nState:\tZ' in status.read_text():break
        time.sleep(1)
    else:
        send(a.pipeline,signal.SIGCONT);send(a.queue,signal.SIGCONT)
        raise RuntimeError('Trainer did not exit; parents released and no probe launched')
    send(a.queue,signal.SIGKILL);send(a.pipeline,signal.SIGKILL)
    record('checkpoint_preserved_training_stopped',checkpoint_sha256=h.hexdigest())
    passed=False
    try:
        cmd=[sys.executable,'scripts/probe_distill40_prefetch.py','--checkpoint','/root/autodl-tmp/checkpoints/smolvla_libero',
             '--data','/root/autodl-tmp/datasets/libero40_distill_86958911','--resume',str(ckpt),'--output',str(audit)]
        with (audit/'probe.log').open('a') as f:subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
        result=json.loads((audit/'result.json').read_text())
        passed=result.get('passed') is True and result['runs']['prefetch']['process_peak_rss_kib']<60*2**20
        record('probe_finished',passed=passed,speedup=result['speedup'])
    except Exception as e:record('probe_failed',error=repr(e))
    # A failed probe returns to the unchanged batch16 path from the saved state.
    if passed:pipeline_argv+=['--cpu-prefetch']
    with (root/'stage1.log').open('a') as f:
        new=subprocess.Popen(pipeline_argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
    queue_argv[queue_argv.index('--pipeline-pid')+1]=str(new.pid)
    with (root/'s5_formal_queue.log').open('a') as f:
        queue=subprocess.Popen(queue_argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
    record('restarted',pipeline_pid=new.pid,queue_pid=queue.pid,cpu_prefetch=passed,resume_update=1000)
    code=new.wait();record('pipeline_exit',returncode=code)
    queue.wait()

if __name__=='__main__':main()
