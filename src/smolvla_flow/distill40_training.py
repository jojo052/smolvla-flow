"""Pure-teacher update and complete action-expert recovery state."""
import random
import os
import torch
from .distillation import compute_distillation_loss


def shared_prefix_velocity(student, teacher_velocity):
    """Reuse frozen prefix KV only; denoising uses the student's expert."""
    velocity = teacher_velocity.__class__.__new__(teacher_velocity.__class__)
    torch.nn.Module.__init__(velocity)
    velocity.model = student.model
    velocity.prefix_pad_masks = teacher_velocity.prefix_pad_masks
    velocity.past_key_values = teacher_velocity.past_key_values
    return velocity


def validate_shared_prefix_models(teacher, student):
    for name in ('state_proj', 'vlm_with_expert.vlm'):
        left = teacher.model.get_submodule(name)
        right = student.model.get_submodule(name)
        if any(p.requires_grad for p in left.parameters()) or any(p.requires_grad for p in right.parameters()):
            raise ValueError('Shared prefix requires frozen '+name)
        a, b = left.state_dict(), right.state_dict()
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError('Shared prefix weights differ: '+name)


def teacher_only_loss(student, teacher, noise, context, config):
    # Deliberately no dataset action argument in this interface.
    return compute_distillation_loss(student, teacher, noise, context, config=config)


def optimizer_update(parameters, optimizer, losses, *, microbatch_size=1, effective_batch=8):
    """Accumulate mean-reduced microbatch losses for exactly eight samples."""
    if effective_batch not in (8,16) or microbatch_size not in (1,2,4,8,16) or effective_batch % microbatch_size:
        raise ValueError('Microbatch size must divide effective batch eight')
    accumulation_steps = effective_batch // microbatch_size
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    totals = [0.0, 0.0, 0.0]
    count = 0
    for output in losses:
        count += 1
        if count > accumulation_steps:
            raise ValueError('Effective batch must equal eight')
        (output.loss / accumulation_steps).backward()
        for i, value in enumerate((output.loss, output.trajectory_consistency_loss,
                                   output.action_regression_loss)):
            totals[i] += float(value.detach()) / accumulation_steps
    if count != accumulation_steps:
        raise ValueError('Effective batch must equal eight')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError('Non-finite updated parameters')
    return dict(loss=totals[0], trajectory_loss=totals[1], endpoint_loss=totals[2],
                gradient_norm=float(norm), effective_batch=effective_batch)


def save_recovery(path, policy, optimizer, sampler, update, provenance):
    import numpy as np
    names = [n for n, p in policy.named_parameters() if p.requires_grad]
    state = {'trainable': {n:p.detach().cpu().clone() for n,p in policy.named_parameters() if n in names},
             'trainable_names': names, 'optimizer': optimizer.state_dict(),
             'sampler': sampler.state_dict(), 'update': update, 'provenance': provenance,
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def restore_recovery(path, policy, optimizer, sampler, provenance):
    import numpy as np
    state = torch.load(path, map_location='cpu', weights_only=False)
    names = [n for n,p in policy.named_parameters() if p.requires_grad]
    if state['provenance'] != provenance or state['trainable_names'] != names:
        raise ValueError('Recovery provenance or trainable names changed')
    if set(state['trainable']) != set(names):
        raise ValueError('Recovery parameter keys mismatch')
    with torch.no_grad():
        for name,p in policy.named_parameters():
            if name in state['trainable']:
                value=state['trainable'][name]
                if value.shape != p.shape or value.dtype != p.dtype:
                    raise ValueError('Recovery shape/dtype mismatch')
                p.copy_(value)
    optimizer.load_state_dict(state['optimizer'])
    sampler.load_state_dict(state['sampler'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['update']
