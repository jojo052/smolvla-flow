#!/usr/bin/env python3
"""Isolated matched-sample microbatch probe. Never writes formal weights."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from scripts.distill_smolvla_action_expert import _load_policy, _freeze_teacher, _configure_action_only_student, CachedSmolVLAVelocity
from scripts.probe_distill40_update import fingerprint
from smolvla_flow.distill40_training import restore_recovery, optimizer_update, teacher_only_loss
from smolvla_flow.distill40_observations import ObservationReader, processed_observation
from smolvla_flow.distill40_sampling import ObservationSampler
from smolvla_flow.distillation import DistillationConfig


def collate(items):
    result = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            if key in ('observation.language.tokens', 'observation.language.attention_mask'):
                # Preserve token order; appended positions are masked out.
                length = max(v.shape[1] for v in values)
                values = [torch.nn.functional.pad(v, (0, length-v.shape[1]), value=0) for v in values]
            if any(v.shape[1:] != values[0].shape[1:] for v in values):
                raise ValueError(f'Unmatched processed shapes: {key}')
            result[key] = torch.cat(values, dim=0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix-only', action='store_true')
    args = parser.parse_args()
    state = torch.load(args.resume, map_location='cpu', weights_only=False)
    provenance = state['provenance']
    del state
    teacher, tp = _load_policy(args.checkpoint, 'HuggingFaceVLA/smolvla_libero', 10, 'cuda')
    student, sp = _load_policy(args.checkpoint, 'HuggingFaceVLA/smolvla_libero', 5, 'cuda')
    _freeze_teacher(teacher)
    _configure_action_only_student(student)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(.9,.95), eps=1e-8, weight_decay=1e-10)
    manifest = json.loads((args.data/'observation_manifest.json').read_text())
    sampler = ObservationSampler(manifest, 123)
    reader = ObservationReader(args.data, manifest)
    config = DistillationConfig(teacher_steps=10, student_steps=5, action_dim=7)
    frozen = fingerprint(student, True)
    teacher_hash = fingerprint(teacher)
    # Fixed thresholds recorded before trials; this gate is numerical, not a
    # claim that downstream closed-loop success is identical.
    report = {'parameter_atol': 1e-7, 'parameter_rtol': 1e-5,
              'loss_atol': 1e-7, 'loss_rtol': 1e-4, 'trials': []}
    reference = None
    variants = [(1,False),(1,True)] if args.prefix_only else [(1,False),(2,False),(4,False)]
    for size, shared_prefix in variants:
        restore_recovery(args.resume, student, optimizer, sampler, provenance)
        student.train()
        rows = [reader.read(*sampler.sample()) for _ in range(8)]
        noises = [torch.randn(1,50,32,device='cuda') for _ in range(8)]
        teacher_inputs = [processed_observation(row,tp) for row in rows]
        student_inputs = [processed_observation(row,sp) for row in rows]
        input_hash = hashlib.sha256()
        for batch in teacher_inputs + student_inputs:
            for key, value in sorted(batch.items()):
                if isinstance(value, torch.Tensor):
                    input_hash.update(key.encode())
                    input_hash.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        for noise in noises:
            input_hash.update(noise.cpu().numpy().tobytes())
        def losses():
            for offset in range(0,8,size):
                tv = CachedSmolVLAVelocity(teacher,collate(teacher_inputs[offset:offset+size]))
                if shared_prefix:
                    sv = CachedSmolVLAVelocity.__new__(CachedSmolVLAVelocity)
                    torch.nn.Module.__init__(sv)
                    sv.model = student.model
                    sv.prefix_pad_masks = tv.prefix_pad_masks
                    sv.past_key_values = tv.past_key_values
                else:
                    sv = CachedSmolVLAVelocity(student,collate(student_inputs[offset:offset+size]))
                yield teacher_only_loss(sv,tv,torch.cat(noises[offset:offset+size]),torch.zeros(size,1,1,device='cuda'),config)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        metrics = optimizer_update(params,optimizer,losses(),microbatch_size=size)
        torch.cuda.synchronize()
        metrics.update(microbatch=size,shared_prefix=shared_prefix,compute_seconds=time.monotonic()-start,
                       peak_bytes=torch.cuda.max_memory_allocated(), input_sha256=input_hash.hexdigest())
        actual = {n:p.detach().cpu().clone() for n,p in student.named_parameters() if p.requires_grad}
        if reference is None:
            reference = actual
            reference_loss = metrics['loss']
        metrics['max_parameter_error'] = max(float((actual[n]-reference[n]).abs().max()) for n in actual)
        metrics['parameters_close'] = all(torch.allclose(actual[n],reference[n],atol=report['parameter_atol'],rtol=report['parameter_rtol']) for n in actual)
        metrics['loss_close'] = abs(metrics['loss']-reference_loss) <= report['loss_atol']+report['loss_rtol']*abs(reference_loss)
        metrics['frozen_unchanged'] = fingerprint(student,True)==frozen and fingerprint(teacher)==teacher_hash
        report['trials'].append(metrics)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(metrics),flush=True)


if __name__ == '__main__':
    main()
