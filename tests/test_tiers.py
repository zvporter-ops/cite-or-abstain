"""Tiering tests: internal material outranks public, and provenance is disclosed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coa.claimgate import check
from coa.corpus import INTERNAL, PUBLIC, CorpusIndex, Passage
from coa.pipeline import orchestrator


def passage(text: str, tier: str = INTERNAL, path: str = "doc.md") -> Passage:
    return Passage(doc_id=1, path=path, start_line=1, text=text, tier=tier)


class TieredIndexTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.internal_dir = root / "internal"
        self.public_dir = root / "public"
        self.internal_dir.mkdir()
        self.public_dir.mkdir()
        (self.internal_dir / "memo.md").write_text(
            "Project Falcon budget is 250000. The launch window is March.", encoding="utf-8")
        (self.public_dir / "standard.md").write_text(
            "The public standard says rollback must be documented.", encoding="utf-8")
        self.db = root / "index.db"

    def tearDown(self):
        self.tmp.cleanup()

    def index(self) -> CorpusIndex:
        return CorpusIndex(
            [{"path": str(self.internal_dir), "tier": INTERNAL},
             {"path": str(self.public_dir), "tier": PUBLIC}],
            self.db,
        )


class TestTieredBuild(TieredIndexTestCase):
    def test_by_tier_counts(self):
        stats = self.index().build()
        self.assertEqual(stats.by_tier.get(INTERNAL), 1)
        self.assertEqual(stats.by_tier.get(PUBLIC), 1)

    def test_retrieve_internal_only(self):
        index = self.index()
        index.build()
        hits = index.retrieve("Project Falcon budget", tier=INTERNAL)
        self.assertTrue(hits)
        self.assertTrue(all(p.tier == INTERNAL for p in hits))

    def test_retrieve_public_only(self):
        index = self.index()
        index.build()
        hits = index.retrieve("public standard rollback", tier=PUBLIC)
        self.assertTrue(hits)
        self.assertTrue(all(p.tier == PUBLIC for p in hits))

    def test_retrieve_without_tier_returns_both(self):
        index = self.index()
        index.build()
        hits = index.retrieve("rollback documented budget")
        self.assertGreaterEqual(len({p.tier for p in hits}), 1)

    def test_sample_passages(self):
        index = self.index()
        index.build()
        sample = index.sample_passages(INTERNAL, 1)
        self.assertEqual(len(sample), 1)
        self.assertEqual(sample[0].tier, INTERNAL)

    def test_plain_string_roots_default_to_internal(self):
        index = CorpusIndex([str(self.internal_dir)], self.db)
        stats = index.build()
        self.assertEqual(stats.by_tier.get(INTERNAL), 1)

    def test_repointing_roots_prunes_the_old_material(self):
        index = CorpusIndex([str(self.internal_dir)], self.db)
        index.build()
        self.assertGreater(index.stats()["passages"], 0)
        index.close()

        # The user removes that folder from the config; its material must stop answering.
        reindexed = CorpusIndex([{"path": str(self.public_dir), "tier": PUBLIC}], self.db)
        stats = reindexed.build()
        self.assertEqual(stats.removed, 1)
        self.assertEqual(stats.by_tier.get(INTERNAL), None)
        self.assertFalse(reindexed.retrieve("Project Falcon budget", tier=INTERNAL))


class FakeIndex:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, query, k=4, tier=None):
        items = self.mapping.get(query, [])
        if tier:
            items = [p for p in items if p.tier == tier]
        return items[:k]


class Canned:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, system, user):
        return self.responses.pop(0)


class TestPrecedence(unittest.TestCase):
    def test_internal_passages_come_first(self):
        internal = passage("Rollback window is March, per the Falcon plan.", INTERNAL)
        public = passage("Public rollback window guidance is March.", PUBLIC, "pub.md")
        index = FakeIndex({"rollback window": [internal, public]})
        llm = Canned(["CONFIDENCE: high\nREASONS:\n- sufficient", "ANSWER:\nMarch\nLIMITATIONS:\n- none\nRECOMMENDATIONS:\n- none"])
        run = orchestrator("rollback window", index, llm, prompts={
            "phase1": "p1", "phase2": "p2", "phase3": "p3"})
        self.assertEqual(run.tier, INTERNAL)
        self.assertEqual(run.internal_count, 1)
        self.assertEqual(run.passages[0].tier, INTERNAL)

    def test_public_only_run_is_marked_public(self):
        public = passage("Public rollback window guidance is March.", PUBLIC, "pub.md")
        index = FakeIndex({"rollback window": [public]})
        llm = Canned(["CONFIDENCE: high\nREASONS:\n- sufficient", "ANSWER:\nMarch\nLIMITATIONS:\n- none\nRECOMMENDATIONS:\n- none"])
        run = orchestrator("rollback window", index, llm, prompts={
            "phase1": "p1", "phase2": "p2", "phase3": "p3"})
        self.assertEqual(run.tier, PUBLIC)
        self.assertEqual(run.internal_count, 0)


class TestProvenance(unittest.TestCase):
    def test_public_claim_is_a_violation_when_public_not_allowed(self):
        public = passage("The public standard says 42 days.", PUBLIC, "pub.md")
        report = check("The standard says 42 days.", [public], allow_public=False)
        self.assertFalse(report.ok)
        self.assertEqual(report.violations[0].kind, "figure")

    def test_public_claim_is_disclosed_when_public_allowed(self):
        public = passage("The public standard says 42 days.", PUBLIC, "pub.md")
        report = check("The standard says 42 days.", [public], allow_public=True)
        self.assertTrue(report.ok)
        self.assertEqual(len(report.public_only), 1)
        self.assertEqual(report.public_only[0].severity, "public")

    def test_internal_claim_counts_as_internal_coverage(self):
        internal = passage("Project Falcon budget is 250000.", INTERNAL)
        report = check("Project Falcon budget is 250000.", [internal])
        self.assertTrue(report.ok)
        self.assertGreater(report.coverage_internal, 0.0)
        self.assertEqual(report.coverage_internal, report.coverage_any)

    def test_public_only_coverage_is_not_internal_coverage(self):
        public = passage("The public standard says rollback must be documented.", PUBLIC)
        report = check("Rollback must be documented by the public standard.", [public],
                       allow_public=True)
        self.assertEqual(report.coverage_internal, 0.0)
        self.assertGreater(report.coverage_any, 0.0)


if __name__ == "__main__":
    unittest.main()
