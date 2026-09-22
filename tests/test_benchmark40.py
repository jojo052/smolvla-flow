import json
import tempfile
import unittest
from pathlib import Path

from smolvla_flow.benchmark40 import (
    SUITES, atomic_json, episode_key, freeze_manifest, fingerprint,
    latency, markdown_report, summarize, validate_episode, wilson,
)


def episode(model="smolvla_official", suite="libero_spatial", tid=0, idx=0, success=True):
    return {"key": episode_key(model, suite, tid, idx), "run_id": "r", "phase": "formal",
            "model": model, "suite": suite, "task_id": tid, "init_index": idx,
            "init_state_sha256": f"{suite}-{tid}-{idx}", "status": "complete", "success": success,
            "failure_type": None if success else "timeout", "steps": 1 if success else SUITES[suite],
            "elapsed_seconds": 2., "total_seconds": 3., "action_nonfinite_count": 0,
            "select_seconds": [.1] * (1 if success else SUITES[suite]),
            "prediction_seconds": [.09], "peak_vram_bytes": 100}


class Benchmark40Tests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(sum(SUITES.values()) * 100, 132000)
        keys = [episode_key("m", s, t, i) for s in SUITES for t in range(10) for i in range(10)]
        self.assertEqual(len(set(keys)), 400)

    def test_pilot_disjoint(self):
        self.assertNotEqual(episode_key("m", "libero_spatial", 0, 10, "pilot"), episode()["key"])
        for phase, tid, idx in [("formal", 0, 10), ("pilot", 1, 10), ("pilot", 0, 0)]:
            with self.assertRaises(ValueError):
                episode_key("m", "libero_spatial", tid, idx, phase)

    def test_freeze_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "manifest.json"
            self.assertEqual(freeze_manifest(p, {"a": 1}), fingerprint({"a": 1}))
            freeze_manifest(p, {"a": 1})
            with self.assertRaises(ValueError):
                freeze_manifest(p, {"a": 2})
            self.assertEqual(json.loads(p.read_text()), {"a": 1})

    def test_atomic_output(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "a.json"
            atomic_json(p, {"ok": True})
            self.assertTrue(json.loads(p.read_text())["ok"])
            self.assertFalse(p.with_suffix(".json.tmp").exists())
            with self.assertRaises(ValueError):
                atomic_json(p, {"nan": float("nan")})
            self.assertTrue(json.loads(p.read_text())["ok"])

    def test_validation_rejects_identity_errors_and_missing_metrics(self):
        base = episode()
        validate_episode(base, "r", base["key"])
        for field, value in [("status", "error"), ("run_id", "changed"), ("success", "true"),
                             ("elapsed_seconds", float("nan")), ("select_seconds", []),
                             ("prediction_seconds", []), ("action_nonfinite_count", 1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_episode({**base, field: value}, "r", base["key"])

    def test_timeout_preserved_as_valid_completed_episode(self):
        row = episode(success=False)
        validate_episode(row, "r", row["key"])
        with self.assertRaises(ValueError):
            validate_episode({**row, "steps": 3}, "r", row["key"])

    def test_partial_has_no_formal_suite_score(self):
        report = summarize([episode()], ["smolvla_official"])
        row = report["models"]["smolvla_official"]
        self.assertIsNone(row["macro_success_rate"])
        self.assertIsNone(row["suites"]["libero_spatial"]["success_rate"])
        self.assertIn("待完成", markdown_report(report))
        self.assertEqual(len(report["tasks"]), 40)

    def test_pilot_not_counted(self):
        self.assertEqual(summarize([{**episode(), "phase": "pilot"}], ["smolvla_official"])["models"]["smolvla_official"]["trials"], 0)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            summarize([episode(), episode()], ["smolvla_official"])

    def test_fail_time_in_success_throughput(self):
        rows = [episode(), episode(idx=1, success=False)]
        self.assertEqual(summarize(rows, ["smolvla_official"])["models"]["smolvla_official"]["success_per_hour_including_reset"], 600)

    def test_wilson_and_latency(self):
        ci = wilson(10, 10)
        self.assertAlmostEqual(ci[0], .7224672, places=6)
        self.assertAlmostEqual(ci[1], 1)
        self.assertEqual(latency([.001, .003])["p50_ms"], 2)
        self.assertEqual(latency([.001, .003])["p95_ms"], 3)

    def test_complete_paired_and_bootstrap(self):
        rows = [episode(m, s, t, i, m == "a") for m in ["a", "b"] for s in SUITES for t in range(10) for i in range(10)]
        report = summarize(rows, ["a", "b"])
        self.assertEqual(report["paired"]["counts"]["first_only"], 400)
        self.assertEqual(report["paired"]["task_bootstrap_95"], [1., 1.])
        self.assertEqual(report["models"]["a"]["macro_success_rate"], 1)
        rows[-1]["init_state_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            summarize(rows, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
