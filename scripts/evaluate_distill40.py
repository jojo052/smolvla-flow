#!/usr/bin/env python3
"""Independent synchronous S5/S2/T2 evaluation using the frozen rollout body."""
import argparse
import json
import time
import types
import importlib.util
import os
import traceback
from pathlib import Path
import torch
frozen_path = Path(os.environ.get('DISTILL40_FROZEN_RUNNER', str(Path(__file__).with_name('run_libero40.py'))))
import smolvla_flow
import scripts
# Keep benchmark dependencies from its frozen checkout; new distillation modules
# remain available from the current project as fallback package paths.
smolvla_flow.__path__ = [str(frozen_path.parents[1]/'src'/'smolvla_flow')] + list(smolvla_flow.__path__)
scripts.__path__ = [str(frozen_path.parent)] + list(scripts.__path__)
spec = importlib.util.spec_from_file_location('distill40_frozen_runner', frozen_path)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
from scripts.train_distill40_stage import overlay
from scripts.distill_smolvla_action_expert import _configure_action_only_student
from smolvla_flow.benchmark40 import atomic_json, sha256, freeze_manifest
from smolvla_flow.distill40_selection import validate_selection_episodes


def key(model,suite,task,index,phase):
    return f'{phase}/{model}/{suite}/task{task:02d}/init{index:02d}'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--student',type=Path)
    p.add_argument('--model',choices=['T2','S5','S2'],required=True)
    p.add_argument('--assets-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['validation','formal','pilot'],required=True)
    p.add_argument('--acknowledge-errors',action='store_true')
    args=p.parse_args()
    if (args.output/'error.json').exists() and not args.acknowledge_errors:
        raise RuntimeError('Inspect recorded execution error before explicitly resuming')
    if (args.student is None)!=(args.model=='T2'): raise ValueError('Student weights required only for students')
    torch.set_num_threads(4)
    config=json.loads((frozen.ROOT/'configs/libero40_public_v1.json').read_text())
    manifest,suites=frozen.prepare_manifest(args,config)
    for task in manifest['tasks']:
        states=suites[task['suite']].get_task_init_states(task['task_id'])
        task['state_sha256']=[__import__('hashlib').sha256(x.detach().cpu().numpy().tobytes() if isinstance(x,torch.Tensor) else x.tobytes()).hexdigest() for x in states[:13]]
        if len(task['state_sha256'])!=13: raise ValueError('Validation states missing')
    steps=5 if args.model=='S5' else 2
    checkpoint_hash=sha256(args.student) if args.student else manifest['checkpoint_files']['model.safetensors']
    manifest['distillation_evaluation']={'model':args.model,'steps':steps,'student_sha256':checkpoint_hash,
        'phase':args.phase,'entry_sha256':sha256(Path(__file__))}
    manifest['protocol']=dict(manifest['protocol'],flow_steps=steps)
    run_id=freeze_manifest(args.output/'manifest.json',manifest)
    from scripts.run_libero_rollout import parse_args, _load_policy
    loaded=_load_policy(parse_args(['--checkpoint',str(args.checkpoint),'--flow-steps',str(steps),'--disable-rtc']),action_execution_steps=1)
    if args.student:
        names=_configure_action_only_student(loaded.policy)
        overlay(loaded.policy,args.student,names)
    loaded.policy.eval()
    for parameter in loaded.policy.parameters(): parameter.requires_grad_(False)
    from lerobot.envs.libero import LiberoEnv
    class Env(LiberoEnv):
        def step(self,action):
            raw,reward,done,info=self._env.step(action)
            success=bool(self._env.check_success())
            return self._format_raw_obs(raw),reward,bool(done or success),False,{**info,'is_success':success}
    # Isolated globals let the unchanged rollout implementation accept validation
    # indices without weakening the frozen benchmark's episode-key validator.
    globals_copy=dict(frozen.run_episode.__globals__)
    globals_copy['episode_key']=lambda _model,suite,task,index,phase:key(args.model,suite,task,index,phase)
    rollout=types.FunctionType(frozen.run_episode.__code__,globals_copy)
    indices=[11,12] if args.phase=='validation' else ([10] if args.phase=='pilot' else list(range(10)))
    rows=[]
    for task in manifest['tasks']:
        if args.phase=='pilot' and task['task_id']!=0: continue
        env=None
        try:
            for index in indices:
                relative=key(args.model,task['suite'],task['task_id'],index,args.phase)
                path=args.output/(relative+'.json')
                if path.exists():
                    row=json.loads(path.read_text())
                    if row['run_id']!=run_id or row['key']!=relative or row['status']!='complete': raise ValueError('Existing episode mismatch')
                else:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    if env is None:
                        env=Env(task_suite=suites[task['suite']],task_id=task['task_id'],task_suite_name=task['suite'],episode_length=task['max_steps'],obs_type='pixels_agent_pos',observation_width=256,observation_height=256,num_steps_wait=10,init_states=True,n_envs=1,control_mode='relative')
                    row=rollout(env,loaded,task,index,args.phase,run_id,{k:path.parent/f'first_{k}.mp4' for k in ('success','failure')})
                    row.update(model=args.model,checkpoint_sha256=checkpoint_hash)
                    atomic_json(path,row)
                if row['init_state_sha256']!=task['state_sha256'][index] or row['checkpoint_sha256']!=checkpoint_hash: raise ValueError('Episode provenance mismatch')
                rows.append(row)
                print(relative,row['success'],flush=True)
        except Exception:
            atomic_json(args.output/'error.json',{'traceback':traceback.format_exc(),'time':time.time()})
            raise
        finally:
            if env is not None: env.close()
    expected=80 if args.phase=='validation' else (4 if args.phase=='pilot' else 400)
    if len(rows)!=expected: raise ValueError('Incomplete evaluation')
    if args.phase=='validation': validate_selection_episodes(rows,checkpoint_hash)
    atomic_json(args.output/'result.json',{'complete':True,'checkpoint_sha256':checkpoint_hash,'episodes':rows,'successes':sum(r['success'] for r in rows)})

if __name__=='__main__': main()
