import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from smolvla_flow.distill40_training import optimizer_update, save_recovery, restore_recovery, validate_shared_prefix_models


class Sampler:
    def state_dict(self): return {'count': 0}
    def load_state_dict(self, state): self.state = state


class RecoveryTests(unittest.TestCase):
    def test_shared_prefix_rejects_changed_or_trainable_backbone(self):
        import copy
        model = torch.nn.Module()
        model.state_proj = torch.nn.Linear(2, 2)
        model.vlm_with_expert = torch.nn.Module()
        model.vlm_with_expert.vlm = torch.nn.Linear(2, 2)
        model.requires_grad_(False)
        teacher = SimpleNamespace(model=model)
        student = SimpleNamespace(model=copy.deepcopy(model))
        validate_shared_prefix_models(teacher, student)
        student.model.state_proj.weight.requires_grad_(True)
        with self.assertRaises(ValueError): validate_shared_prefix_models(teacher, student)
        student.model.state_proj.weight.requires_grad_(False)
        with torch.no_grad(): student.model.state_proj.weight.add_(1)
        with self.assertRaises(ValueError): validate_shared_prefix_models(teacher, student)

    def test_microbatch_mean_weighting(self):
        x = torch.arange(16, dtype=torch.float64).reshape(8, 2) / 16
        expected = None
        for size in (1, 2, 4, 8):
            model = torch.nn.Linear(2, 1, dtype=torch.float64)
            with torch.no_grad():
                model.weight.fill_(0.1)
                model.bias.fill_(0.2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            def losses():
                for batch in x.split(size):
                    loss = model(batch).square().mean()
                    yield SimpleNamespace(loss=loss, trajectory_consistency_loss=loss, action_regression_loss=loss)
            optimizer_update(model.parameters(), optimizer, losses(), microbatch_size=size)
            actual = {n:p.detach().clone() for n,p in model.named_parameters()}
            if expected is None:
                expected = actual
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-12)

    def test_exact_continuation(self):
        torch.manual_seed(123)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9,.95), eps=1e-8, weight_decay=1e-10)
        def update():
            def losses():
                for _ in range(8):
                    loss = model(torch.randn(1,2)).square().mean()
                    yield SimpleNamespace(loss=loss, trajectory_consistency_loss=loss, action_regression_loss=loss)
            return optimizer_update(model.parameters(), optimizer, losses())
        update()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recovery.pt'
            save_recovery(path,model,optimizer,Sampler(),1,{'test':True})
            expected_metrics=update()
            expected={n:p.detach().clone() for n,p in model.named_parameters()}
            self.assertEqual(restore_recovery(path,model,optimizer,Sampler(),{'test':True}),1)
            self.assertEqual(expected_metrics,update())
            for name,p in model.named_parameters():
                torch.testing.assert_close(p,expected[name],rtol=0,atol=0)

if __name__=='__main__': unittest.main()
