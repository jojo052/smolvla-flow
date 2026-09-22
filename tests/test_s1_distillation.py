import torch
from torch import nn
from smolvla_flow.distillation import DistillationConfig,compute_distillation_loss,build_coarse_teacher_targets
from smolvla_flow.distill40_training import teacher_only_loss


class Velocity(nn.Module):
    def __init__(self,value):
        super().__init__();self.value=nn.Parameter(torch.tensor(float(value)))
    def forward(self,x,t,context):return self.value.expand_as(x)


def test_two_to_one_target_keeps_endpoints():
    states=torch.tensor([1.,.7,.2]).reshape(1,3,1,1)
    assert torch.equal(build_coarse_teacher_targets(states,1),states[:,[0,2]])


def test_s1_pure_teacher_loss_and_gradient():
    teacher=Velocity(.5);student=Velocity(.2)
    noise=torch.randn(2,50,32);context=torch.zeros(2,1,1)
    out=teacher_only_loss(student,teacher,noise,context,DistillationConfig(teacher_steps=2,student_steps=1,action_dim=7))
    torch.testing.assert_close(out.loss,1.5*out.action_loss)
    torch.testing.assert_close(out.trajectory_loss,.5*out.action_loss)
    out.loss.backward()
    assert teacher.value.grad is None and student.value.grad is not None


def test_pure_teacher_interface_rejects_expert_actions():
    import inspect
    assert 'target_actions' not in inspect.signature(teacher_only_loss).parameters


def test_candidate_selection_uses_samples_not_update_rate():
    from scripts.run_s1_pipeline import choose_probe
    a=dict(passed=True,provenance={'batch':16},samples_per_second=100)
    b=dict(passed=True,provenance={'batch':32},samples_per_second=104.9)
    assert choose_probe([a,b]) is a
    b['samples_per_second']=105
    assert choose_probe([a,b]) is b
    b['passed']=False
    assert choose_probe([a,b]) is a


def test_candidate_selection_rejects_all_failed():
    import pytest
    from scripts.run_s1_pipeline import choose_probe
    with pytest.raises(ValueError,match='No eligible'):
        choose_probe([{'passed':False}])


def test_expert_actions_do_not_enter_processed_inputs():
    import numpy as np
    from smolvla_flow.distill40_observations import processed_observation,ObservationReader
    row={'task':'fixed task','observation.state':[0.]*8,
         'observation.images.image':np.zeros((256,256,3),dtype=np.uint8),
         'observation.images.image2':np.zeros((256,256,3),dtype=np.uint8)}
    reference=processed_observation(row,lambda x:x,decoded=True)
    for action in (np.zeros((50,7)),np.full((50,7),12345.)):
        actual=processed_observation(dict(row,action=action),lambda x:x,decoded=True)
        assert actual.keys()==reference.keys()
        for key in reference:
            if isinstance(reference[key],torch.Tensor):
                torch.testing.assert_close(reference[key],actual[key],rtol=0,atol=0)
            else:
                assert reference[key]==actual[key]
    assert 'action' not in ObservationReader.columns
