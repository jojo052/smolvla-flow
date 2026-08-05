import torch

from smolvla_flow.action_expert import (
    ActionExpertConfig,
    TransformerActionExpert,
    make_action_flow_batch,
    sample_action_chunk,
)


def small_config() -> ActionExpertConfig:
    return ActionExpertConfig(
        action_dim=3,
        chunk_size=8,
        context_dim=12,
        model_dim=32,
        time_dim=8,
        num_layers=2,
        num_heads=4,
        feedforward_dim=64,
    )


def test_action_flow_batch_shapes() -> None:
    actions = torch.randn(4, 8, 3)
    noisy_actions, time, velocity = make_action_flow_batch(actions)
    assert noisy_actions.shape == velocity.shape == actions.shape
    assert time.shape == (4, 1)


def test_action_expert_forward_and_backward() -> None:
    model = TransformerActionExpert(small_config())
    actions = torch.randn(4, 8, 3)
    noisy_actions, time, target = make_action_flow_batch(actions)
    context = torch.randn(4, 5, 12)
    prediction = model(noisy_actions, time, context)
    loss = (prediction - target).square().mean()
    loss.backward()
    assert prediction.shape == actions.shape
    assert model.velocity_head.weight.grad is not None


def test_action_chunk_sampler_shape() -> None:
    model = TransformerActionExpert(small_config()).eval()
    context = torch.randn(2, 5, 12)
    initial_noise = torch.randn(2, 8, 3)
    output = sample_action_chunk(model, context, num_steps=2, initial_noise=initial_noise)
    assert output.shape == (2, 8, 3)


def test_action_chunk_sampler_uses_descending_time() -> None:
    class RecordingExpert(TransformerActionExpert):
        def __init__(self) -> None:
            super().__init__(small_config())
            self.times: list[torch.Tensor] = []

        def forward(self, noisy_actions, time, context_tokens):
            del context_tokens
            self.times.append(time.detach().clone())
            return torch.ones_like(noisy_actions)

    model = RecordingExpert().eval()
    context = torch.zeros(1, 2, 12)
    initial_noise = torch.zeros(1, 8, 3)
    output = sample_action_chunk(model, context, num_steps=2, initial_noise=initial_noise)
    torch.testing.assert_close(torch.cat(model.times, dim=1), torch.tensor([[1.0, 0.5]]))
    torch.testing.assert_close(output, torch.full_like(output, -1.0))


def test_action_expert_validates_chunk_shape() -> None:
    model = TransformerActionExpert(small_config())
    try:
        model(torch.randn(2, 7, 3), torch.rand(2, 1), torch.randn(2, 5, 12))
    except ValueError as error:
        assert "trailing shape" in str(error)
    else:
        raise AssertionError("expected ValueError")
