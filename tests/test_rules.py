"""Correction ledger tests: a correction becomes a rule, and repetition makes it hard.

The point of the ledger is that the user never has to say the same thing three times.
Detection therefore has to be conservative -- a false positive silently injects a rule
the user never stated, which is worse than missing one.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coa.rules import RuleStore, detect_correction, render_injection, render_notice


class TestDetection(unittest.TestCase):
    def test_a_plain_correction_is_captured(self):
        text = detect_correction("no, use plain language and drop the hedging footer")
        self.assertIsNotNone(text)
        self.assertIn("plain language", text)

    def test_i_told_you_is_a_correction(self):
        self.assertIsNotNone(detect_correction("I told you to keep the answers short"))

    def test_an_explicit_standing_instruction_is_captured(self):
        self.assertIsNotNone(detect_correction("always run the tests before you say done"))

    def test_from_now_on_is_captured(self):
        text = detect_correction("from now on, write in plain english and no bullet lists")
        self.assertIsNotNone(text)

    def test_instead_is_captured(self):
        self.assertIsNotNone(detect_correction("use the vault copy instead of re-fetching"))

    def test_a_question_is_not_a_correction(self):
        self.assertIsNone(detect_correction("What is the Project Falcon budget?"))

    def test_a_question_that_starts_with_dont_is_not_a_correction(self):
        self.assertIsNone(detect_correction("don't you think that's right?"))

    def test_a_bare_acknowledgment_is_not_a_correction(self):
        self.assertIsNone(detect_correction("thanks, that worked"))

    def test_a_short_no_is_not_a_correction(self):
        self.assertIsNone(detect_correction("no, ok"))

    def test_an_ordinary_statement_is_not_a_correction(self):
        self.assertIsNone(detect_correction("The Project Falcon budget is 250000."))

    def test_empty_input_is_not_a_correction(self):
        self.assertIsNone(detect_correction(""))
        self.assertIsNone(detect_correction(None))

    def test_leading_filler_is_stripped_from_the_rule(self):
        text = detect_correction("No, actually, always cite the passage id when you answer")
        self.assertIsNotNone(text)
        self.assertFalse(text.lower().startswith("no"))
        self.assertIn("cite the passage id", text)

    def test_the_rule_is_bounded(self):
        text = detect_correction("always " + "keep going " * 200)
        self.assertIsNotNone(text)
        self.assertLessEqual(len(text), 240)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "rules.json"
        self.store = RuleStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_capture_creates_a_rule(self):
        outcome = self.store.add("use plain language", session="s1")
        self.assertTrue(outcome.created)
        self.assertFalse(outcome.escalated)
        self.assertEqual(outcome.rule["id"], 1)
        self.assertFalse(outcome.rule["hard"])
        self.assertEqual(len(self.store.rules()), 1)

    def test_the_same_correction_twice_escalates_to_hard(self):
        first = self.store.add("use plain language and drop the hedging footer")
        second = self.store.add("use plain language and drop the hedging footer")
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertTrue(second.escalated)
        self.assertEqual(second.rule["id"], first.rule["id"])
        self.assertEqual(second.rule["hits"], 2)
        self.assertTrue(second.rule["hard"])
        self.assertEqual(len(self.store.rules()), 1)

    def test_a_paraphrased_correction_is_the_same_rule(self):
        self.store.add("use plain language and drop the hedging footer")
        outcome = self.store.add("drop the hedging footer, use plain language")
        self.assertFalse(outcome.created)
        self.assertTrue(outcome.escalated)

    def test_a_different_correction_is_a_new_rule(self):
        self.store.add("use plain language")
        outcome = self.store.add("always run the tests before saying done")
        self.assertTrue(outcome.created)
        self.assertEqual(outcome.rule["id"], 2)
        self.assertEqual(len(self.store.rules()), 2)

    def test_rules_survive_a_reload(self):
        self.store.add("use plain language")
        reopened = RuleStore(self.path)
        self.assertEqual(len(reopened.rules()), 1)
        self.assertEqual(reopened.rules()[0]["text"], "use plain language")

    def test_forget_removes_a_rule(self):
        outcome = self.store.add("use plain language")
        self.assertTrue(self.store.forget(outcome.rule["id"]))
        self.assertEqual(self.store.rules(), [])

    def test_forget_of_an_unknown_id_is_false(self):
        self.assertFalse(self.store.forget(99))

    def test_mark_hard_promotes_a_rule(self):
        outcome = self.store.add("use plain language")
        self.assertTrue(self.store.set_hard(outcome.rule["id"]))
        self.assertTrue(self.store.rules()[0]["hard"])

    def test_ids_do_not_repeat_after_forget(self):
        first = self.store.add("use plain language")
        self.store.forget(first.rule["id"])
        second = self.store.add("always run the tests before saying done")
        self.assertNotEqual(second.rule["id"], first.rule["id"])


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RuleStore(Path(self.tmp.name) / "rules.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_empty_store_injects_nothing(self):
        self.assertEqual(render_injection(self.store.rules()), "")

    def test_injection_names_the_rules_and_their_precedence(self):
        self.store.add("use plain language and drop the hedging footer")
        block = render_injection(self.store.rules())
        self.assertIn("STANDING RULES", block)
        self.assertIn("plain language", block)
        self.assertIn("outrank", block)

    def test_hard_rules_are_marked(self):
        self.store.add("use plain language")
        self.store.add("use plain language")
        block = render_injection(self.store.rules())
        self.assertIn("[hard]", block)

    def test_injection_is_bounded(self):
        for i in range(40):
            self.store.add(f"always keep rule number {i} in mind when writing anything long")
        block = render_injection(self.store.rules(), max_chars=400)
        self.assertLessEqual(len(block), 400)

    def test_a_new_rule_notice_says_it_was_saved(self):
        outcome = self.store.add("use plain language")
        notice = render_notice(outcome)
        self.assertIn("saved", notice.lower())
        self.assertIn("plain language", notice)

    def test_an_escalation_notice_says_it_is_hard_now(self):
        self.store.add("use plain language")
        outcome = self.store.add("use plain language")
        notice = render_notice(outcome)
        self.assertIn("hard", notice.lower())
        self.assertIn("stated before", notice.lower())


if __name__ == "__main__":
    unittest.main()
