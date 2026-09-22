#!/usr/bin/env python3
"""Read-only model probe before admitting a new parallel LIBERO executor.

This does not run formal episodes or modify any checkpoint. Tolerances are
fixed before measurement. A failed batch gate does not authorize a rollout.
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.evaluate_distill40 import frozen
from scripts.run_libero_rollout import _load_policy, parse_args, _make_observation_pipeline
from scripts.train_distill40_stage import overlay
from scripts.distill_smolvla_action_expert import _configure_action_only_student
from smolvla_flow.benchmark40 import atomic_json, sha256

RTOL, ATOL = 1e-5, 1e-6


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    return value


def stack(rows):
    if any(set(row) != set(rows[0]) for row in rows):
        raise ValueError('Processor keys changed')
    result = {}
    for k in rows[0]:
        if not k.startswith('observation.'):
            continue  # Transition metadata is unused by the action predictor.
        values = [row[k] for row in rows]
        if k in ('observation.language.tokens', 'observation.language.attention_mask'):
            length = max(x.shape[1] for x in values)
            # Right padding adds only masked tokens, never truncates a sample.
            values = [torch.nn.functional.pad(x, (0, length-x.shape[1]), value=0) for x in values]
        result[k] = torch.cat(values, dim=0)
    return result


def compare(actual, reference):
    error = (actual - reference).abs()
    limits = ATOL + RTOL * reference.abs()
    return {'passed': bool(torch.isfinite(actual).all() and (error <= limits).all()),
            'max_abs': error.max().item(), 'mse': error.square().mean().item(),
            'outside_tolerance': int((error > limits).sum()), 'elements': error.numel()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--assets-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(123)
    config = json.loads((frozen.ROOT / 'configs/libero40_public_v1.json').read_text())
    manifest, suites = frozen.prepare_manifest(args, config)
    student_hash = sha256(args.student)
    if student_hash != '61ae24f990e5cb290e9dedaac5d82daabd6e4e3910449a09b1b13d001f52fe81':
        raise ValueError('Selected S5 identity changed')
    manifest.update(probe_source_sha256=sha256(Path(__file__)), student_sha256=student_hash,
                    tolerance={'rtol': RTOL, 'atol': ATOL}, formal_rollouts=False)
    atomic_json(args.output / 'manifest.json', manifest)
    loaded = _load_policy(parse_args(['--checkpoint', str(args.checkpoint), '--flow-steps', '1',
                                     '--disable-rtc']), action_execution_steps=1)
    policy = loaded.policy
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    from lerobot.envs.libero import LiberoEnv
    observations, identities = [], []
    for task in manifest['tasks']:
        if task['suite'] != 'libero_10':
            continue
        env = LiberoEnv(task_suite=suites[task['suite']], task_id=task['task_id'],
                       task_suite_name=task['suite'], episode_length=task['max_steps'],
                       obs_type='pixels_agent_pos', observation_width=256, observation_height=256,
                       num_steps_wait=10, init_states=True, n_envs=1, control_mode='relative')
        try:
            for index in (0, 1):
                env.init_state_id = index
                obs, _ = env.reset(seed=123)
                assert env.init_state_id == index + 1 and env.num_steps_wait == 10
                batch = _make_observation_pipeline(task['language'])(obs, loaded.preprocessor)
                observations.append(batch)
                identities.append({'task': task, 'init_index': index})
        finally:
            env.close()
    generator = torch.Generator(device='cuda').manual_seed(123)
    noises = [torch.normal(0., 1., size=(1, 50, 32), device='cuda', generator=generator)
              for _ in range(80)]
    bank = args.output / 'inputs.pt'
    torch.save({'observations': [cpu_tree(x) for x in observations],
                'noise': [x.cpu() for x in noises], 'identities': identities}, bank)
    atomic_json(args.output / 'inputs.json', {'file_sha256': sha256(bank), 'observations': 20,
                'noises_per_observation': 4, 'shape': [1, 50, 32], 'identities': identities})
    outputs, counts = {}, {}
    original = policy.model.denoise_step
    times = []
    def counted(*a, **kw):
        timestep = kw.get('timestep', a[3] if len(a) > 3 else None)
        times.append(float(timestep.flatten()[0]))
        return original(*a, **kw)
    policy.model.denoise_step = counted
    with torch.inference_mode():
        for name, steps in [('T10', 10), ('T5', 5), ('T2', 2), ('T1', 1), ('S5', 5)]:
            if name == 'S5':
                names = _configure_action_only_student(policy)
                overlay(policy, args.student, names)
                policy.eval()
                for p in policy.parameters():
                    p.requires_grad_(False)
            policy.config.num_steps = policy.model.config.num_steps = steps
            policy.reset()
            times.clear()
            chunk = policy._get_action_chunk(observations[0], noise=noises[0].clone())
            grid = list(times)
            assert len(grid) == steps
            assert all(abs(t - (1-i/steps)) < 1e-6 for i, t in enumerate(grid))
            assert chunk.shape == (1, 50, 7) and torch.isfinite(chunk).all()
            policy.reset()
            action = policy.select_action(observations[0], noise=noises[0].clone())
            torch.testing.assert_close(action, chunk[:, 0], rtol=RTOL, atol=ATOL)
            policy.reset()
            counts[name] = {'denoise_calls': len(grid), 'times': grid, 'dt': -1/steps,
                            'end': 0., 'select_action_aligned': True}
            atomic_json(args.output / 'integration.json', counts)
            print('INTEGRATION', name, counts[name], flush=True)
            # T1 batch gate uses 32 independently processed observations/noises.
            if name == 'T1':
                rows = [observations[i % 20] for i in range(32)]
                ns = noises[:32]
                refs = torch.cat([policy._get_action_chunk(b, noise=n.clone())
                                  for b, n in zip(rows, ns)])
                gates = []
                for size in (1, 2, 4, 8, 16, 32):
                    for reverse in (False, True):
                        order = list(range(32))[::(-1 if reverse else 1)]
                        parts = []
                        for start in range(0, 32, size):
                            ids = order[start:start+size]
                            parts.append(policy._get_action_chunk(stack([rows[i] for i in ids]),
                                         noise=torch.cat([ns[i] for i in ids])))
                        actual = torch.cat(parts)
                        reference = refs[order]
                        result = {'batch': size, 'reverse': reverse, **compare(actual, reference)}
                        gates.append(result)
                        print('BATCH_GATE', result, flush=True)
                        atomic_json(args.output / 'batch_gate.json', {'rtol': RTOL, 'atol': ATOL,
                                    'results': gates, 'formal_authorized': False})
            if name in ('T10', 'T5', 'S5'):
                values = []
                for i, obs in enumerate(observations):
                    for j in range(4):
                        policy.reset()
                        values.append(policy._get_action_chunk(obs, noise=noises[4*i+j].clone()).cpu())
                outputs[name] = torch.cat(values)
                torch.save(outputs, args.output / 'actions.pt')
        policy.model.denoise_step = original
    diagnosis = {}
    for name in ('T5', 'S5'):
        err = (outputs[name]-outputs['T10']).square()
        diagnosis[name] = {'first_action_mse': err[:, 0].mean().item(),
                           'full_chunk_mse': err.mean().item(),
                           'per_position_mse': err.mean(dim=(0, 2)).tolist(),
                           'translation_mse': err[:, :, :3].mean().item(),
                           'rotation_mse': err[:, :, 3:6].mean().item(),
                           'gripper_mse': err[:, :, 6].mean().item(),
                           'per_observation_mse': err.reshape(20, 4, 50, 7).mean(dim=(1, 2, 3)).tolist()}
    atomic_json(args.output / 'diagnosis.json', {'normalized_action_units': True, 'models': diagnosis,
                'actions_sha256': sha256(args.output / 'actions.pt'),
                'does_not_measure_closed_loop_success': True})
    print('PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
