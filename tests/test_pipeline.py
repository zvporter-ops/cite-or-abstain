"""Pipeline tests.

Each test encodes a defect the pipeline exists to prevent. The abstention test is
written so that a Phase-3 call would raise: the canned response list has no fourth
entry, so "it abstained" cannot be true if generation happened.
"""

from __future__ import annotations

import unittest

from coa.corpus import Passage
from coa.pipeline import orchestrator, parse_answer, parse_context_needed, render


def passage(text: str, path: str = "notes.md", line: int = 1) -> Passage:
    return Passage(doc_id=1, path=path, start_line=line, text=text)


class FakeIndex:
    """Returns passages for exact query strings; unknown queries retrieve nothing."""

    def __init__(self, mapping: dict[str, list[Passage]]):
        self.mapping = mapping
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int = 4, tier: str | None = None) -> list[Passage]:
        self.queries.append(query)
        items = self.mapping.get(query, [])
        if tier:
            items = [p for p in items if p.tier == tier]
        return list(items)[:k]


class Canned:
    """Records prompts and returns canned responses in order; extra calls raise."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.systems: list[str] = []
        self.users: list[str] = []

    def __call__(self, system: str, user: str) -> str:
        self.systems.append(system)
        self.users.append(user)
        if not self.responses:
            raise IndexError("the pipeline made more model calls than expected")
        return self.responses.pop(0)


LOW_ASSESSMENT = """CONFIDENCE: low
REASONS:
- the context does not contain a rollback procedure
- no deployment history is available
- the release notes would help"""

MEDIUM_ASSESSMENT = """CONFIDENCE: medium
REASONS:
- the context covers the deployment
- rollback steps are thin
- a runbook would help"""

ANSWER = """ANSWER:
Restore the previous release from the backup, then redeploy.
LIMITATIONS:
- no verification that the backup is current
RECOMMENDATIONS:
- add a post-rollback check"""


class TestAbstention(unittest.TestCase):
    def test_low_confidence_abstains_without_generating(self):
        index = FakeIndex({})
        llm = Canned([
            LOW_ASSESSMENT,
            "CONTEXT_NEEDED:\n- rollback procedure\n- deployment history",
            # No third response: a Phase-3 call would raise IndexError.
        ])
        run = orchestrator("how do I roll back a deploy", index, llm)
        self.assertTrue(run.abstained)
        self.assertEqual(run.confidence, "low")
        self.assertIn("Abstained", run.answer)
        self.assertEqual(len(llm.responses), 0)

    def test_abstention_returns_retrieval_targets_as_recommendations(self):
        index = FakeIndex({})
        llm = Canned([
            LOW_ASSESSMENT,
            "CONTEXT_NEEDED:\n- rollback procedure\n- deployment history",
        ])
        run = orchestrator("how do I roll back a deploy", index, llm)
        self.assertEqual(run.recommendations, ["rollback procedure", "deployment history"])

    def test_empty_corpus_never_generates(self):
        index = FakeIndex({})
        llm = Canned([LOW_ASSESSMENT, "CONTEXT_NEEDED:\n- anything"])
        run = orchestrator("what is the soil mix for tomatoes", index, llm)
        self.assertTrue(run.abstained)
        self.assertEqual(run.answer.count("ANSWER:"), 0)


class TestPhaseTwoRescue(unittest.TestCase):
    def test_phase_two_rescue_promotes_and_answers(self):
        # Phase 1 rates low over nothing; Phase 2 retrieves material that covers the
        # whole query, so the re-gate must promote and Phase 3 must run.
        rich = passage(
            "Rollback procedure: restore the previous release from backup, "
            "then redeploy the deployment."
        )
        index = FakeIndex({"rollback procedure": [rich]})
        llm = Canned([
            LOW_ASSESSMENT,
            "CONTEXT_NEEDED:\n- rollback procedure",
            ANSWER,
        ])
        run = orchestrator("how do I roll back a deploy", index, llm)
        self.assertFalse(run.abstained)
        self.assertEqual(run.confidence, "medium")
        self.assertIn("Restore the previous release", run.answer)
        self.assertIn("rollback procedure", index.queries)

    def test_regate_scores_retrieved_context_not_initial_context(self):
        rich = passage("Rollback from backup, redeploy the deployment.")
        index = FakeIndex({"rollback procedure": [rich]})
        llm = Canned([LOW_ASSESSMENT, "CONTEXT_NEEDED:\n- rollback procedure", ANSWER])
        run = orchestrator("how do I roll back a deploy", index, llm)
        gate_phase = [p for p in run.phases if p.get("phase") == "gate"][0]
        self.assertEqual(gate_phase["initial_overlap"], 0.0)
        self.assertGreater(gate_phase["final_overlap"], 0.0)

    def test_retrieved_passages_are_cited(self):
        rich = passage("Rollback from backup, redeploy the deployment.", path="runbook.md")
        index = FakeIndex({"rollback procedure": [rich]})
        llm = Canned([LOW_ASSESSMENT, "CONTEXT_NEEDED:\n- rollback procedure", ANSWER])
        run = orchestrator("how do I roll back a deploy", index, llm)
        self.assertIn("runbook.md:1", [p.cite() for p in run.passages])


class TestConfidenceParsing(unittest.TestCase):
    def test_template_echo_is_not_returned(self):
        # A model echoing the template must not become a confidence value.
        llm = Canned(["CONFIDENCE: [high|medium|low]\nREASONS:\n- insufficient context",
                      "CONTEXT_NEEDED:\n- x"])
        index = FakeIndex({})
        run = orchestrator("anything", index, llm, prompts={
            "phase1": "p1", "phase2": "p2", "phase3": "p3"})
        self.assertNotIn(run.self_reported, ("high|medium|low",))
        self.assertEqual(run.self_reported, "low")

    def test_high_rating_over_thin_context_is_demoted(self):
        llm = Canned(["CONFIDENCE: high\nREASONS:\n- looks complete",
                      "CONTEXT_NEEDED:\n- nothing", ANSWER])
        index = FakeIndex({})
        run = orchestrator("tomato soil mix", index, llm, prompts={
            "phase1": "p1", "phase2": "p2", "phase3": "p3"})
        self.assertEqual(run.confidence, "medium")
        self.assertFalse(run.abstained)


class TestDomainPrompts(unittest.TestCase):
    def test_domain_prompts_are_actually_used(self):
        llm = Canned([LOW_ASSESSMENT, "CONTEXT_NEEDED:\n- x"])
        index = FakeIndex({})
        orchestrator("review this", index, llm, domain="code_review")
        self.assertIn("reviewer", llm.systems[0])
        self.assertNotIn("research analyst", llm.systems[0])

    def test_unknown_domain_raises(self):
        with self.assertRaises(KeyError):
            orchestrator("q", FakeIndex({}), Canned([]), domain="nope")


class TestParsing(unittest.TestCase):
    def test_parse_answer_splits_sections(self):
        answer, limits, recs = parse_answer(ANSWER)
        self.assertIn("Restore the previous release", answer)
        self.assertEqual(limits, ["no verification that the backup is current"])
        self.assertEqual(recs, ["add a post-rollback check"])

    def test_parse_context_needed_handles_missing_marker(self):
        self.assertEqual(parse_context_needed("no marker here"), [])

    def test_render_respects_char_budget(self):
        passages = [passage("x" * 1_000, line=i) for i in range(10)]
        out = render(passages, max_chars=2_500, per_passage=1_000)
        self.assertLessEqual(len(out), 2_600)


if __name__ == "__main__":
    unittest.main()
