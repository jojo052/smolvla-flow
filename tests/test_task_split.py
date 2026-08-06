from smolvla_flow.task_split import count_full_windows, split_episode_indices


def test_count_full_windows_matches_distillation_stride() -> None:
    assert count_full_windows(49) == 0
    assert count_full_windows(50) == 1
    assert count_full_windows(84) == 9
    assert count_full_windows(84, stride=50) == 1


def test_split_is_deterministic_disjoint_and_exhaustive() -> None:
    indices = list(range(1272, 1317))
    first = split_episode_indices(indices, seed=0)
    second = split_episode_indices(indices, seed=0)
    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 31,
        "validation": 7,
        "test": 7,
    }
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert sorted(first["train"] + first["validation"] + first["test"]) == indices
