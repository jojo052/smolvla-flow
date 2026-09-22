"""Fixed-input latency diagnosis only. Never produces formal success-rate rows."""
import argparse
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def render_process(connection,stop,assets,task):
    """A spawned process owns its only environment and EGL context, no policy."""
    import traceback
    try:
        import torch
        import numpy as np
        from scripts import run_libero_rollout
        from scripts.evaluate_s1_interleaved import make_env
        torch.set_num_threads(1)
        run_libero_rollout._configure_libero(assets)
        import libero.libero as core
        from libero.libero import benchmark
        core._assets_path_cache=str(assets)
        env=make_env(benchmark.get_benchmark_dict()[task['suite']](),task)
        try:
            env.init_state_id=13;obs,_=env.reset(seed=123);connection.send(('ready',obs))
            while True:
                command,hold=connection.recv()
                if command=='quit':break
                if command=='reset':
                    env.init_state_id=13;obs,_=env.reset(seed=123);connection.send(('ready',obs));continue
                if command!='render':raise ValueError(command)
                start=time.perf_counter();steps=0;env_seconds=0.;late=[]
                connection.send(('started',None))
                while not stop.is_set():
                    deadline=start+steps/20
                    if stop.wait(max(0.,deadline-time.perf_counter())):break
                    late.append(max(0.,time.perf_counter()-deadline))
                    before=time.perf_counter();env.step(np.asarray(hold,dtype=np.float32))
                    env_seconds+=time.perf_counter()-before;steps+=1
                connection.send(('done',dict(steps=steps,env_seconds=env_seconds,late=late)))
        finally:env.close()
    except BaseException:connection.send(('error',traceback.format_exc()))
    finally:connection.close()


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','student','assets-dir','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split-process',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new diagnostic output')
    import numpy as np
    import torch
    from scripts import run_libero_rollout
    from scripts.s1_rtc_policy import Policy
    from scripts.evaluate_s1_interleaved import make_env
    from scripts.evaluate_s1_rtc import relative_control_evidence
    from smolvla_flow.benchmark40 import atomic_json,sha256
    from scripts.report_s1_rtc import distribution
    from libero.libero import benchmark
    torch.set_num_threads(4)
    run_libero_rollout._configure_libero(a.assets_dir)
    import libero.libero as core
    core._assets_path_cache=str(a.assets_dir)
    manifest=json.loads(a.manifest.read_text())
    if sha256(a.student)!=manifest['student_sha256']:raise ValueError('Student hash mismatch')
    task=next(t for t in manifest['tasks'] if t['suite']=='libero_10' and t['task_id']==5)
    suite=benchmark.get_benchmark_dict()['libero_10']()
    policy=Policy(a);env=make_env(suite,task)
    a.output.mkdir(parents=True)
    records=[];summaries=[];child=None;connection=None;env_closed=False;split_observation_equal=None
    try:
        env.init_state_id=13;obs,_=env.reset(seed=123)
        evidence=relative_control_evidence(env)
        fixed=copy.deepcopy(obs)
        noise=torch.randn((1,50,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))
        prime=policy.predict(copy.deepcopy(fixed),task['language'],'async',0,noise=noise.clone())
        prefix=prime['normalized'][10:]
        hold=np.array([0.]*6+[prime['actions'][0][6]],dtype=np.float32)
        references={}
        def receive(expected):
            if not connection.poll(90):raise TimeoutError('Render worker timeout')
            tag,value=connection.recv()
            if tag!=expected:raise RuntimeError(f'Render worker {tag}: {value}')
            return value
        if a.split_process:
            import multiprocessing as mp
            env.close();env_closed=True
            context=mp.get_context('spawn');connection,other=context.Pipe();stop=context.Event()
            child=context.Process(target=render_process,args=(other,stop,a.assets_dir,task));child.start();other.close()
            split_obs=receive('ready')
            def equal(x,y):
                if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
                return bool(np.array_equal(x,y))
            split_observation_equal=equal(fixed,split_obs)
            if not split_observation_equal:raise ValueError('Spawned environment initial observation differs')
        def predict(mode):
            started=time.perf_counter()
            result=policy.predict(copy.deepcopy(fixed),task['language'],mode,0,
                prefix if mode=='rtc' else None,2,noise.clone())
            return result,time.perf_counter()-started,torch.get_num_threads()
        with ThreadPoolExecutor(max_workers=1) as pool:
            for mode in ('async','rtc'):
                for condition in ('main_static','worker_static','process_render20hz' if a.split_process else 'worker_render20hz'):
                    if a.split_process:connection.send(('reset',None));receive('ready')
                    else:env.init_state_id=13;env.reset(seed=123)
                    for index in range(18):
                        start=time.perf_counter();steps=0;env_seconds=0.;late=[]
                        if condition=='main_static':result,elapsed,threads=predict(mode)
                        else:
                            if condition=='process_render20hz':
                                stop.clear();connection.send(('render',hold.tolist()));receive('started')
                            future=pool.submit(predict,mode)
                            if condition=='worker_render20hz':
                                while not future.done():
                                    deadline=start+steps/20
                                    time.sleep(max(0.,deadline-time.perf_counter()))
                                    if future.done():break
                                    late.append(max(0.,time.perf_counter()-deadline))
                                    before=time.perf_counter();env.step(hold);env_seconds+=time.perf_counter()-before;steps+=1
                            result,elapsed,threads=future.result()
                            if condition=='process_render20hz':
                                stop.set();rendered=receive('done')
                                steps=rendered['steps'];env_seconds=rendered['env_seconds'];late=rendered['late']
                        output=np.asarray(result['normalized'])
                        if mode not in references:references[mode]=output.copy()
                        delta=float(np.max(np.abs(output-references[mode])))
                        record=dict(mode=mode,condition=condition,index=index,warmup=index<3,
                            call_seconds=elapsed,intraop_threads=threads,output_max_abs_delta=delta,
                            allclose=bool(np.allclose(output,references[mode],rtol=1e-5,atol=1e-6)),
                            render_steps=steps,environment_seconds=env_seconds,max_lateness=max(late,default=0),
                            **result['metrics'])
                        records.append(record)
                        atomic_json(a.output/'progress.json',dict(last=record,completed=len(records)))
                    chosen=[r for r in records if r['mode']==mode and r['condition']==condition and not r['warmup']]
                    summary=dict(mode=mode,condition=condition,n=len(chosen),
                        timings={k:distribution(r[k] for r in chosen) for k in
                            ('call_seconds','prediction_s','denoise_forward_s','rtc_extra_s','preprocess_s','postprocess_s')},
                        intraop_threads=sorted({r['intraop_threads'] for r in chosen}),
                        output_allclose=all(r['allclose'] for r in chosen))
                    summaries.append(summary);print(json.dumps(summary),flush=True)
        atomic_json(a.output/'diagnosis.json',dict(student_sha256=sha256(a.student),
            entry_sha256=sha256(Path(__file__)),controller=evidence,summaries=summaries,requests=records,
            split_process=a.split_process,split_observation_equal=split_observation_equal,
            limitation='Fixed observations/noise; render uses zero motion. No formal task scores or protocol change.'))
    finally:
        if not env_closed:env.close()
        if child is not None:
            stop.set()
            if child.is_alive():
                connection.send(('quit',None));child.join(timeout=10)
            if child.is_alive():child.terminate();child.join()
            connection.close()


if __name__=='__main__':
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    main()
