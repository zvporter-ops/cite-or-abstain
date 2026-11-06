"""Scorecard tests: the battery must measure, not flatter.

Gate probes are deterministic, so they are tested exactly. Pipeline probes need a model, so
they run against a scripted one that behaves the way a model should -- and against ones that
misbehave, which is the case that matters.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coa.audit import build_index, gate_probes, render, run_audit
from coa.corpus import INTERNAL, PUBLIC, Passage, content_words

ANSWER = "ANSWER:\nMarch\nLIMITATIONS:\n- none\nRECOMMENDATIONS:\n- none"


class ScriptedModel:
    """Behaves by phase: high confidence when the context has content, low when it does not."""

    def __init__(self):
        self.calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        if "Name what you need to look up" in system:
            return "CONTEXT_NEEDED:\n- project falcon budget\n- launch window"
        if "Answer the question using only the context" in system:
            return ANSWER
        if "(no context retrieved)" in user:
            return "CONFIDENCE: low\nREASONS:\n- no context\n"
        return "CONFIDENCE: high\nREASONS:\n- context present\n"


class FakeIndex:
    def __init__(self, internal: list[Passage]):
        self.internal = internal

    def sample_passages(self, tier=INTERNAL, n=3):
        return self.internal[:n]

    def stats(self):
        return {"documents": 1, "passages": len(self.internal),
                "by_tier": {INTERNAL: len(self.internal)}, "sources": []}

    def retrieve(self, query, k=4, tier=None):
        # Answer only what the passage actually covers; everything else is silence.
        if not self.internal:
            return []
        wanted = set(content_words(query))
        have = set(content_words(self.internal[0].text))
        return [self.internal[0]] if wanted & have else []


def passage(text: str) -> Passage:
    return Passage(doc_id=1, path="memo.md", start_line=1, text=text, tier=INTERNAL)


MEMO = passage("Project Falcon budget is 250000. The launch window is March 14 to March 28.")


class TestGateProbes(unittest.TestCase):
    def test_probes_catch_and_allow(self):
        checks, question = gate_probes(FakeIndex([MEMO]))
        self.assertTrue(checks["invented_quote_caught"])
        self.assertTrue(checks["invented_figure_caught"])
        self.assertTrue(checks["verbatim_quote_allowed"])
        self.assertTrue(checks["paraphrase_allowed"])
        self.assertIn("project falcon", (question or "").lower())

    def test_no_internal_material_skips_gate_probes(self):
        checks, question = gate_probes(FakeIndex([]))
        self.assertEqual(checks, {})
        self.assertIsNone(question)


class TestAuditRun(unittest.TestCase):
    def test_well_behaved_deployment_is_trusted(self):
        report = run_audit(FakeIndex([MEMO]), ScriptedModel(), relevance_floor=0.6)
        self.assertEqual(report["score"], 100)
        self.assertEqual(report["verdict"], "trusted")
        self.assertEqual(report["pipeline"]["abstained_on_unanswerable"], "2/2")
        self.assertEqual(report["pipeline"]["answered_known_question"], "1/1")

    def test_pipeline_alone_cannot_reach_trusted(self):
        # Gate results are half the score. With no internal passage to probe, a perfect
        # pipeline still cannot round up to "trusted".
        report = run_audit(FakeIndex([]), ScriptedModel())
        self.assertEqual(report["gate"], {})
        self.assertEqual(report["score"], 30)
        self.assertEqual(report["verdict"], "untrusted")

    def test_endpoint_failure_still_reports_gate_results(self):
        def broken(system: str, user: str) -> str:
            raise ConnectionError("endpoint unreachable")

        report = run_audit(FakeIndex([MEMO]), broken)
        self.assertIn("ConnectionError", report["pipeline"]["model_error"])
        self.assertTrue(report["gate"]["invented_figure_caught"])
        self.assertLess(report["score"], 85)

    def test_clearance_is_reported(self):
        report = run_audit(FakeIndex([MEMO]), ScriptedModel(), local=False)
        self.assertEqual(report["clearance"]["session_model"], "remote")
        self.assertIn("withheld", report["clearance"]["internal_material"])
        self.assertIn("remote model", render(report))

    def test_no_internal_material_is_untrusted(self):
        report = run_audit(FakeIndex([]), ScriptedModel())
        self.assertEqual(report["gate"], {})
        self.assertLess(report["score"], 60)
        self.assertEqual(report["verdict"], "untrusted")


class TestBuildIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.internal = root / "internal"
        self.public = root / "public"
        self.internal.mkdir()
        self.public.mkdir()
        (self.internal / "memo.md").write_text("Falcon budget is 250000.", encoding="utf-8")
        (self.public / "handbook.md").write_text("Rollback window definition.", encoding="utf-8")
        self.db = root / "index.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_builds_from_plain_settings(self):
        index = build_index({"corpus_roots": [str(self.internal)],
                             "public_roots": [str(self.public)]}, self.db)
        assert index is not None
        stats = index.stats()
        self.assertEqual(stats["by_tier"].get(INTERNAL), 1)
        self.assertEqual(stats["by_tier"].get(PUBLIC), 1)

    def test_no_sources_is_none(self):
        self.assertIsNone(build_index({}, self.db))


if __name__ == "__main__":
    unittest.main()
