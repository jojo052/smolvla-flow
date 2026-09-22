"""Artifact-backed regression checks, CPU only."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.audit_three_policy_results import audit

ROOTS = {m: Path('artifacts') / p for m, p in [('T10', 'libero40_public_v1'), ('T5', 'libero40_t5_direct_v1'), ('S5', 'libero40_s5_b16_v1')]}


@unittest.skipUnless(all((p / 'manifest.json').exists() for p in ROOTS.values()), 'Requires downloaded result artifacts')
class AuditTest(unittest.TestCase):
    def test_complete(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            audit(ROOTS, Path(d))
            pairs = json.loads((Path(d) / 'paired_episodes.json').read_text())
            self.assertEqual(len(pairs), 400)
            self.assertEqual(sum(p['S5_T5_category'] == 'T5_only' for p in pairs), 21)

    def corrupt(self, field, value):
        original = Path.read_text
        def read(p, *a, **kw):
            text = original(p, *a, **kw)
            if 'libero40_t5_direct_v1/formal' in str(p) and p.name == 'init00.json':
                row = json.loads(text); row[field] = value
                return json.dumps(row)
            return text
        with tempfile.TemporaryDirectory() as d, patch.object(Path, 'read_text', read), self.assertRaises((AssertionError, ValueError)):
            audit(ROOTS, Path(d))

    def test_wrong_state_rejected(self):
        self.corrupt('init_state_sha256', 'wrong')

    def test_wrong_checkpoint_rejected(self):
        self.corrupt('checkpoint_sha256', 'wrong')

    def test_incomplete_rejected(self):
        self.corrupt('status', 'error')
