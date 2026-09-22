import torch

from scripts.build_baseline_data import _window_episode


def _rows(length: int) -> list[dict[str, object]]:
    return [
        {
            "episode_index": 3,
            "frame_index": frame,
            "task_index": 34,
            "observation.state": [float(frame)] * 8,
            "action": [float(frame)] * 7,
        }
        for frame in range(length)
    ]


def test_window_export_keeps_state_and_action_contract() -> None:
    state, actions, starts = _window_episode(
        _rows(80),
        chunk_size=50,
        horizon=64,
        stride=4,
    )
    assert state.shape == (5, 8)
    assert actions.shape == (5, 64, 7)
    assert starts == [0, 4, 8, 12, 16]
    torch.testing.assert_close(state[0], torch.zeros(8))
    torch.testing.assert_close(actions[1, 0], torch.full((7,), 4.0))


def test_window_export_skips_short_episode() -> None:
    state, actions, starts = _window_episode(
        _rows(63),
        chunk_size=50,
        horizon=64,
        stride=4,
    )
    assert state.shape == (0, 8)
    assert actions.shape == (0, 64, 7)
    assert starts == []
