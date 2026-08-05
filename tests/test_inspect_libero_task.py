from smolvla_flow.libero_mapping import (
    aggregate_action_dimension,
    match_dataset_task_index,
    normalize_task_text,
)


def test_normalize_task_text_handles_libero_style_names() -> None:
    assert normalize_task_text("Pick_up_the_black_bowl.") == "pick up the black bowl"


def test_match_dataset_task_index_with_dict_metadata() -> None:
    tasks = {
        "12": {"task": "open the middle drawer"},
        "27": {"task": "pick up the black bowl"},
    }
    assert match_dataset_task_index("Pick up the black bowl.", tasks) == 27


def test_aggregate_action_dimension_combines_episode_moments() -> None:
    episodes = [
        {
            "stats/action/count": [2],
            "stats/action/min": [0, 0, 0, 0, 0, 0, -1],
            "stats/action/max": [0, 0, 0, 0, 0, 0, 1],
            "stats/action/mean": [0, 0, 0, 0, 0, 0, -0.5],
            "stats/action/std": [0, 0, 0, 0, 0, 0, 0.5],
        },
        {
            "stats/action/count": [2],
            "stats/action/min": [0, 0, 0, 0, 0, 0, -1],
            "stats/action/max": [0, 0, 0, 0, 0, 0, 1],
            "stats/action/mean": [0, 0, 0, 0, 0, 0, 0.5],
            "stats/action/std": [0, 0, 0, 0, 0, 0, 0.5],
        },
    ]
    stats = aggregate_action_dimension(episodes, dimension=6)
    assert stats["count"] == 4
    assert stats["mean"] == 0
    assert stats["std"] == 2**-0.5
    assert stats["observed_both_hysteresis_sides"] is True
