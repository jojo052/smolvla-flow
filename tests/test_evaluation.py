from smolvla_flow.evaluation import evaluate_acceptance


def _rollout(mode: str, *, success: bool) -> dict:
    return {
        "mode": mode,
        "dataset_task_index": 34,
        "torch_seed": 123,
        "episodes": [
            {
                "seed": 0,
                "torch_seed": 123,
                "success": success,
                "steps": 10,
                "waiting_ticks": 0,
                "effective_control_hz": 15.0,
                "action_finite_ratio": 1.0,
                "action_smoothness_mean": 0.1 if mode == "async" else 0.2,
                "chunk_boundary_jump_mean": 0.1 if mode == "async" else 0.2,
                "inference_events": [
                    {"deadline_miss": False},
                    {"deadline_miss": False},
                ],
            },
            {
                "seed": 1,
                "torch_seed": 123,
                "success": success,
                "steps": 10,
                "waiting_ticks": 0,
                "effective_control_hz": 15.0,
                "action_finite_ratio": 1.0,
                "action_smoothness_mean": 0.1 if mode == "async" else 0.2,
                "chunk_boundary_jump_mean": 0.1 if mode == "async" else 0.2,
                "inference_events": [
                    {"deadline_miss": False},
                    {"deadline_miss": False},
                ],
            },
        ],
    }


def test_evaluate_acceptance_reports_pass_fail_and_unknown() -> None:
    sync = _rollout("sync", success=True)
    asynchronous = _rollout("async", success=False)
    benchmark = {
        "teacher": {"mean_seconds": 1.0, "output_finite": True},
        "student": {"mean_seconds": 0.4, "output_finite": True},
    }

    result = evaluate_acceptance(
        [sync, asynchronous],
        [benchmark],
        fused_rollout=asynchronous,
        unfused_rollout=sync,
    )

    assert result["finite_output"]["status"] == "pass"
    assert result["task_isolation"]["status"] == "pass"
    assert result["waiting_ticks"]["status"] == "pass"
    assert result["deadline_miss_ratio"]["status"] == "pass"
    assert result["effective_control_frequency"]["status"] == "pass"
    assert result["two_step_speedup"]["status"] == "pass"
    assert result["async_success_drop"]["status"] == "fail"
    assert result["chunk_boundary_jump"]["status"] == "pass"
    assert result["same_episodes_and_seeds"]["status"] == "pass"
    assert result["validation_action_error"]["status"] == "unknown"


def test_evaluate_acceptance_does_not_infer_missing_fields() -> None:
    result = evaluate_acceptance([{"mode": "async", "episodes": [{"seed": 0, "steps": 1}]}])

    assert result["finite_output"]["status"] == "unknown"
    assert result["task_isolation"]["status"] == "unknown"
    assert result["waiting_ticks"]["status"] == "unknown"
    assert result["deadline_miss_ratio"]["status"] == "unknown"
    assert result["same_episodes_and_seeds"]["status"] == "unknown"
