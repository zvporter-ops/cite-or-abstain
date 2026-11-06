"""Claim gate tests.

The fixtures are adversarial on purpose. A gate that catches an invented quote but
blocks a legitimate paraphrase is not usable, so the paraphrase test matters as much
as the fabrication tests.
"""

from __future__ import annotations

import unittest

from coa.claimgate import check, render_report
from coa.corpus import Passage

RUNBOOK = Passage(
    doc_id=1, path="runbook.md", start_line=10,
    text=("Rollback procedure: restore the previous release from backup, then "
          "redeploy the deployment. The window is 30 minutes."),
)


class TestQuotes(unittest.TestCase):
    def test_invented_quote_is_a_violation(self):
        answer = 'The runbook says "restore from the oldest snapshot available" first.'
        report = check(answer, [RUNBOOK])
        self.assertFalse(report.ok)
        self.assertEqual([f.kind for f in report.violations], ["quote"])

    def test_verbatim_quote_passes(self):
        answer = 'It says "restore the previous release from backup" and then redeploy.'
        report = check(answer, [RUNBOOK])
        self.assertTrue(report.ok, report.to_dict())

    def test_curly_quote_is_checked(self):
        answer = "It says \u201cthe oldest snapshot available\u201d first."
        report = check(answer, [RUNBOOK])
        self.assertFalse(report.ok)

    def test_quote_with_reflowed_whitespace_passes(self):
        answer = 'It says "restore the previous\nrelease from backup".'
        report = check(answer, [RUNBOOK])
        self.assertTrue(report.ok, report.to_dict())


class TestFigures(unittest.TestCase):
    def test_invented_figure_is_a_violation(self):
        answer = "The rollback took 47% longer than expected."
        report = check(answer, [RUNBOOK])
        self.assertFalse(report.ok)
        self.assertEqual([f.kind for f in report.violations], ["figure"])

    def test_figure_present_in_evidence_passes(self):
        answer = "The window is 30 minutes."
        report = check(answer, [RUNBOOK])
        self.assertTrue(report.ok, report.to_dict())

    def test_invented_date_is_a_violation(self):
        answer = "This changed on 2026-03-14."
        report = check(answer, [RUNBOOK])
        self.assertFalse(report.ok)
        self.assertIn("date", [f.kind for f in report.violations])

    def test_invented_url_is_a_violation(self):
        answer = "See https://example.com/rollback-guide for details."
        report = check(answer, [RUNBOOK])
        self.assertFalse(report.ok)
        self.assertIn("url", [f.kind for f in report.violations])

    def test_invented_identifier_is_a_violation(self):
        answer = "Run `deploy --force-rollback` to recover."
        report = check(answer, [RUNBOOK])
        self.assertFalse(report.ok)
        self.assertIn("identifier", [f.kind for f in report.violations])


class TestParaphrase(unittest.TestCase):
    def test_legitimate_paraphrase_is_not_blocking(self):
        answer = ("Restore the previous release from backup and then redeploy. "
                  "Teams usually find this safer than a hotfix.")
        report = check(answer, [RUNBOOK])
        self.assertTrue(report.ok, report.to_dict())
        self.assertTrue(any(f.severity == "soft" for f in report.findings))

    def test_proper_noun_is_soft_only(self):
        answer = "The SRE Runbook Committee owns this procedure."
        report = check(answer, [RUNBOOK])
        self.assertTrue(report.ok)
        self.assertIn("name", [f.kind for f in report.findings])


class TestNoEvidence(unittest.TestCase):
    def test_claims_without_evidence_are_violations(self):
        answer = "Roughly 42% of deploys fail within 10 minutes."
        report = check(answer, [])
        self.assertFalse(report.ok)
        self.assertGreaterEqual(len(report.violations), 1)

    def test_empty_answer_is_not_an_error(self):
        report = check("", [RUNBOOK])
        self.assertTrue(report.ok)
        self.assertEqual(report.coverage, 0.0)


class TestReport(unittest.TestCase):
    def test_coverage_counts_supported_sentences(self):
        answer = ("Restore the previous release from backup, then redeploy the deployment. "
                  "Unrelated sentence about gardening tomatoes.")
        report = check(answer, [RUNBOOK])
        self.assertGreater(report.coverage, 0.0)
        self.assertLess(report.coverage, 1.0)

    def test_render_report_names_violations(self):
        report = check("It took 47% longer.", [RUNBOOK])
        text = render_report(report)
        self.assertIn("unsupported claim", text)
        self.assertIn("47%", text)

    def test_render_report_clean_case(self):
        report = check("The window is 30 minutes.", [RUNBOOK])
        text = render_report(report)
        self.assertIn("No unsupported claims", text)


class TestCitationNoise(unittest.TestCase):
    def test_line_references_are_not_figures(self):
        report = check("See lines 5 and 11-12 of the memo (12 lines).", [RUNBOOK])
        self.assertFalse([f for f in report.violations if f.kind == "figure"])

    def test_real_figures_are_still_caught(self):
        report = check("The budget was 918,442 units.", [RUNBOOK])
        self.assertTrue([f for f in report.violations if f.kind == "figure"])


if __name__ == "__main__":
    unittest.main()
