"""Approved initial-wait correction for T10, independent artifact version."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

from smolvla_flow.benchmark40 import atomic_json, sha256, freeze_manifest, fingerprint
from smolvla_flow.rtc_benchmark import PROTOCOL, validate_row


def schedule(phase):
    if phase not in ('check', 'formal'):
        raise ValueError(phase)
    pairs = [(s, t, i) for s in PROTOCOL['suites']
             for t in ([0] if phase == 'check' else [0, 5])
             for i in ([13] if phase == 'check' else range(27, 32))]
    random.Random(123).shuffle(pairs)
    conditions = [('async', 0), ('rtc', 0), ('async', 100), ('rtc', 100)]
    jobs = []
    for n, (suite, task, init) in enumerate(pairs):
        for mode, delay in conditions[n % 4:] + conditions[:n % 4]:
            jobs.append(dict(phase=phase, suite=suite, task_id=task, init_index=init,
                mode=mode, delay_ms=delay,
                key=f'{phase}/{suite}/task{task:02d}/init{init:02d}/{mode}_d{delay}'))
    return jobs


def worker(connection, args):
    try:
        import torch
        import resource
        from scripts.t10_rtc_policy import Policy
        torch.set_num_threads(4)
        policy = Policy(args)
        connection.send((0, True, 'ready'))
        while True:
            seq, method, pos, kw = connection.recv()
            if method == 'close':
                break
            if method not in ('predict', 'reset', 'preflight'):
                raise ValueError('Unknown RPC')
            start = time.perf_counter()
            result = getattr(policy, method)(*pos, **kw)
            if method == 'predict':
                result['metrics'].update(guidance_weight_cap=1.25, model_id='official_T10',
                    worker_call_s=time.perf_counter()-start,
                    worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            connection.send((seq, True, result))
    except BaseException:
        connection.send((locals().get('seq', 0), False, traceback.format_exc()))
    finally:
        connection.close()


def validate_identity(row, job, manifest, run_id):
    validate_row(row, job, run_id)
    task = next(t for t in manifest['tasks'] if (t['suite'], t['task_id']) == (job['suite'], job['task_id']))
    if row['init_state_sha256'] != task['state_sha256'][job['init_index']]:
        raise ValueError('State mismatch')
    if row.get('model_id') != 'official_T10' or 'student_sha256' in row:
        raise ValueError('Unexpected student identity')
    if row['checkpoint_files_fingerprint'] != fingerprint(manifest['checkpoint_files']):
        raise ValueError('Weight identity mismatch')
    for q in row['requests']:
        if q['forward_calls'] != 10 or q['rtc_calls'] != (10 if q['rtc_active'] else 0):
            raise ValueError('Ten-step call count mismatch')
        if q['guidance_weight_cap'] != 1.25:
            raise ValueError('Strength drift')


def report(output):
    from scripts.report_s1_rtc import distribution
    manifest = json.loads((output/'manifest.json').read_text())
    run_id = fingerprint(manifest)
    if (output/'error.json').exists():
        raise ValueError('Unresolved error')
    expected = {output/'episodes'/f"{j['key']}.json" for p in ('check', 'formal') for j in schedule(p)}
    if set((output/'episodes').rglob('*.json')) != expected:
        raise ValueError('Need exactly 16 checks and 160 formal records')
    if not json.loads((output/'preflight.json').read_text())['passed']:
        raise ValueError('Missing preflight')
    rows = []
    for phase in ('check', 'formal'):
        for job in schedule(phase):
            r = json.loads((output/'episodes'/f"{job['key']}.json").read_text())
            validate_identity(r, job, manifest, run_id)
            if r.get('video') and not (output/r['video']).exists():
                raise ValueError('Missing video')
            if phase == 'formal':
                rows.append(r)
    groups = []
    comparisons = []
    for delay in (0, 100):
        for mode in ('async', 'rtc'):
            rr = [r for r in rows if r['mode'] == mode and r['delay_ms'] == delay]
            ticks = [t for r in rr for t in r['ticks']]
            requests = [q for r in rr for q in r['requests']]
            wins = sum(r['success'] for r in rr)
            sec = sum(r['total_seconds'] for r in rr)
            jumps = [t['chunk_boundary_jump'] for t in ticks if t['chunk_boundary_jump']]
            groups.append(dict(mode=mode, delay_ms=delay, success=wins, episodes=len(rr),
                success_rate=wins/len(rr), total_seconds=sec, success_per_hour=3600*wins/sec,
                episode_seconds=distribution(r['total_seconds'] for r in rr),
                prediction_seconds=distribution(q['prediction_s'] for q in requests),
                queue_empty_fraction=sum(t['waiting'] for t in ticks)/len(ticks),
                late_fraction=sum(t['lateness_s'] > .005 for t in ticks)/len(ticks),
                jumps={k: distribution(x[k] for x in jumps) for k in ('translation', 'rotation', 'gripper')}))
        maps = [{(r['suite'], r['task_id'], r['init_index']): r for r in rows
                 if r['mode'] == mode and r['delay_ms'] == delay} for mode in ('rtc', 'async')]
        if len(maps[0]) != 40 or maps[0].keys() != maps[1].keys():
            raise ValueError('Incomplete pairing')
        counts = dict(both_success=0, first_only=0, second_only=0, both_fail=0)
        deltas = []
        for suite in PROTOCOL['suites']:
            for task in (0, 5):
                ds = []
                for init in range(27, 32):
                    a, b = [m[suite, task, init] for m in maps]
                    if a['init_state_sha256'] != b['init_state_sha256']:
                        raise ValueError('Unpaired states')
                    x, y = a['success'], b['success']
                    counts['both_success' if x and y else 'first_only' if x else 'second_only' if y else 'both_fail'] += 1
                    ds.append(int(x)-int(y))
                deltas.append(sum(ds)/5)
        rng = random.Random(123)
        boot = sorted(sum(rng.choices(deltas, k=8))/8 for _ in range(10000))
        comparisons.append(dict(delay_ms=delay, first='rtc', second='async', counts=counts,
            difference=sum(deltas)/8, task_bootstrap_95=[boot[249], boot[9749]]))
    atomic_json(output/'report.json', dict(complete=True, episodes=160, checks=16,
        run_id=run_id, groups=groups, paired_comparisons=comparisons,
        limitations=['Eight tasks; approved reuse of T2 initial indices 27-31 after viewing its results; exploratory.',
                    'Cannot directly attribute differences from S1 runs on other states to distillation.']))


def main():
    p = argparse.ArgumentParser()
    for key in ('output', 'checkpoint', 'assets-dir', 'source', 't2-source'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--stage', choices=('run', 'resume', 'report'), default='run')
    args = p.parse_args()
    if args.stage == 'report':
        report(args.output)
        return
    if args.stage == 'run' and args.output.exists():
        raise ValueError('Independent output required')
    if (args.output/'error.json').exists():
        raise ValueError('Inspect recorded error; no automatic retries')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if any(x.strip().isdigit() for x in active.splitlines()):
        raise ValueError('GPU occupied')
    source = json.loads((args.source/'manifest.json').read_text())
    if source['protocol'] != PROTOCOL:
        raise ValueError('Source protocol drift')
    for path, digest in source['entry_sources'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Code drift: '+path)
    for rel, digest in source['checkpoint_files'].items():
        if sha256(args.checkpoint/rel) != digest:
            raise ValueError('Checkpoint/processor drift: '+rel)
    t2_manifest=json.loads((args.t2_source/'manifest.json').read_text())
    if t2_manifest['checkpoint_files']!=source['checkpoint_files'] or t2_manifest['jobs']!=schedule('formal'):
        raise ValueError('T2 weight or paired schedule mismatch')
    if not json.loads((args.t2_source/'report.json').read_text())['complete']:
        raise ValueError('Missing completed T2 reference')
    import torch
    from scripts import run_libero_rollout as loader
    from scripts.evaluate_s1_interleaved import make_env
    from scripts.rtc_initial_wait_controller import episode, state_hash
    from scripts.s1_rtc_process import ProcessPolicy
    from scripts.s1_resources import ResourceLog
    torch.set_num_threads(1)
    loader._configure_libero(args.assets_dir)
    import libero.libero as core
    from libero.libero import benchmark
    core._assets_path_cache = str(args.assets_dir)
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in PROTOCOL['suites']}
    tasks = source['tasks']
    for task in tasks:
        states = suites[task['suite']].get_task_init_states(task['task_id'])
        if len(states) < 32:
            raise ValueError('Missing initial states 27-31')
        hashes = [state_hash(s) for s in states[:32]]
        if hashes[:25] != task['state_sha256']:
            raise ValueError('State drift')
        task['state_sha256'] = hashes
    if tasks!=t2_manifest['tasks']:
        raise ValueError('T2 task/state pairing mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = dict(protocol=dict(PROTOCOL, flow_steps=10, guidance_weight=1.25,
                    initial_wait_policy='no_rejection_before_control_clock; uncapped_measured_delay_estimate',
                    initial_states=list(range(27, 32)), modes=['async', 'rtc']),
        tasks=tasks, model_id='official_T10', checkpoint_files=source['checkpoint_files'],
        original_manifest_sha256=sha256(args.source/'manifest.json'),
        entry_sources={str(f): sha256(f) for f in (Path(__file__), Path(__file__).with_name('t10_rtc_policy.py'),
            Path(__file__).with_name('rtc_initial_wait_controller.py'),
            Path(__file__).parents[1]/'src/smolvla_flow/rtc_initial_wait.py')},
        inherited_sources=source['entry_sources'], execution_runtime=source['execution_runtime'],
        approved_t2_state_reuse=True,t2_manifest_sha256=sha256(args.t2_source/'manifest.json'),
        jobs=schedule('formal'), checks=schedule('check'))
    run_id = freeze_manifest(args.output/'manifest.json', manifest)
    resources = ResourceLog(args.output)
    policy = None
    try:
        policy = ProcessPolicy(args, worker=worker)
        task = tasks[0]
        env = make_env(suites[task['suite']], task)
        try:
            env.init_state_id = 13
            obs, _ = env.reset(seed=123)
            gate = policy.preflight(obs, task['language'])
        finally:
            env.close()
        atomic_json(args.output/'preflight.json', dict(gate, run_id=run_id))
        if not gate['passed']:
            raise ValueError('Ten-step preflight failed')
        for phase in ('check', 'formal'):
            for job in schedule(phase):
                target = args.output/'episodes'/f"{job['key']}.json"
                if target.exists():
                    validate_identity(json.loads(target.read_text()), job, manifest, run_id)
                    continue
                task = next(t for t in tasks if (t['suite'], t['task_id']) == (job['suite'], job['task_id']))
                row = episode(args, job, task, suites[job['suite']], policy, run_id, resources)
                # The shared controller writes its S1-specific label; replace it before persisting.
                row.pop('student_sha256')
                row.update(model_id='official_T10', checkpoint_files_fingerprint=fingerprint(source['checkpoint_files']))
                validate_identity(row, job, manifest, run_id)
                atomic_json(target, row)
                print('COMPLETE', job['key'], row['success'], flush=True)
        resources.check()
        report(args.output)
    except BaseException:
        atomic_json(args.output/'error.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        if policy is not None:
            policy.close()
        resources.close()


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    main()
