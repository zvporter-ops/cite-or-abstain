"""Correction ledger tests: a correction becomes a rule, and repetition makes it hard.

The point of the ledger is that the user never has to say the same thing three times.
Detection therefore has to be conservative -- a false positive silently injects a rule
the user never stated, which is worse than missing one.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from coa.rules import (
    RuleStore,
    detect_correction,
    extract_rules,
    import_standing,
    provenance_rank,
    render_injection,
    render_notice,
)


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


class TestProvenance(unittest.TestCase):
    """Who said it decides what wins. A note the user wrote earlier yields to a fresh word."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.store = RuleStore(self.root / "rules.json")

    def test_the_rank_ladder(self):
        self.assertGreater(provenance_rank({"provenance": "user_stated"}),
                           provenance_rank({"provenance": "imported"}))
        self.assertGreater(provenance_rank({"provenance": "imported"}),
                           provenance_rank({"provenance": "config"}))
        self.assertEqual(provenance_rank({}), 3)  # an unlabelled rule is treated as stated

    def test_a_constraint_inside_a_rule_is_not_a_reversal(self):
        self.store.add("always answer in plain english, no filler", provenance="imported")
        outcome = self.store.add("always answer in plain english and keep it short")
        self.assertFalse(outcome.created)  # same instruction, so it merges
        self.assertEqual(outcome.rule["provenance"], "user_stated")

    def test_a_reversal_by_the_user_suppresses_the_imported_note(self):
        self.store.add("never use bullet lists in answers", provenance="imported")
        self.store.add("always use bullet lists in answers")
        imported = self.store.rules()[0]
        self.assertEqual(imported["superseded_by"], 2)
        block = render_injection(self.store.rules())
        self.assertIn("always use bullet lists", block)
        self.assertNotIn("never use bullet lists", block)

    def test_a_restatement_keeps_the_higher_ranked_wording(self):
        self.store.add("always keep answers short and plain", provenance="imported")
        outcome = self.store.add("always keep answers short plain and free of filler")
        self.assertEqual(outcome.rule["provenance"], "user_stated")
        self.assertEqual(outcome.rule["text"],
                         "always keep answers short plain and free of filler")

    def test_a_suppressed_rule_comes_back_when_the_winner_is_forgotten(self):
        self.store.add("never use bullet lists in answers", provenance="imported")
        self.store.add("always use bullet lists in answers")
        self.assertEqual(self.store.rules()[0]["superseded_by"], 2)
        self.store.forget(2)
        self.assertNotIn("superseded_by", self.store.rules()[0])
        self.assertIn("never use bullet lists", render_injection(self.store.rules()))

    def test_an_imported_note_never_hardens_by_repetition(self):
        self.store.add("always keep the answers short and plain", provenance="imported")
        outcome = self.store.add("always keep the answers short and plain", provenance="imported")
        self.assertFalse(outcome.escalated)
        self.assertFalse(outcome.rule["hard"])

    def test_the_user_restating_an_imported_note_makes_it_hard(self):
        self.store.add("always keep the answers short and plain", provenance="imported")
        outcome = self.store.add("always keep the answers short and plain")
        self.assertEqual(outcome.rule["provenance"], "user_stated")
        self.assertTrue(outcome.rule["hard"])
        self.assertTrue(outcome.escalated)
        self.assertTrue(outcome.upgraded)
        self.assertIn("already in your standing files", render_notice(outcome))

    def test_stated_rules_are_rendered_before_imported_ones(self):
        self.store.add("always write in plain english for every answer", provenance="imported")
        self.store.add("never use bullet lists when a paragraph will do")
        block = render_injection(self.store.rules())
        self.assertLess(block.index("bullet lists"), block.index("plain english"))
        self.assertIn("[imported]", block)

    def test_an_imported_rule_carries_its_tag(self):
        self.store.add("always run the tests before saying done", provenance="imported")
        self.assertIn("- [imported] always run the tests", render_injection(self.store.rules()))


class TestImport(unittest.TestCase):
    """A cold start is not an empty start: instruction bullets the user already wrote are read."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.store = RuleStore(self.root / "rules.json")

    def _write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_instruction_bullets_are_taken_and_prose_is_not(self):
        text = ("# Conventions\n\n"
                "- Always run the tests before you say done.\n"
                "- The project is called Falcon.\n"
                "* Never push straight to master without a review.\n"
                "\n"
                "This paragraph is prose, not a rule, and must not be imported at all.\n")
        found = extract_rules(text)
        self.assertEqual(found, ["Always run the tests before you say done.",
                                 "Never push straight to master without a review."])

    def test_numbered_bullets_count(self):
        self.assertEqual(extract_rules("1. Always answer in plain english, no filler."),
                         ["Always answer in plain english, no filler."])

    def test_an_overlong_bullet_is_not_a_rule(self):
        self.assertEqual(extract_rules("- Always " + "word " * 60), [])

    def test_each_file_is_imported_once(self):
        path = self._write("SOUL.md", "- Always answer in plain english.\n"
                                      "- Never invent file paths you did not read.\n")
        first = import_standing(self.store, [path])
        self.assertEqual(first, {"imported": 2, "files": 1})
        self.assertEqual(import_standing(self.store, [path]), {"imported": 0, "files": 0})
        self.assertEqual(len(self.store.rules()), 2)

    def test_a_changed_file_is_read_again(self):
        path = self._write("SOUL.md", "- Always answer in plain english.\n")
        import_standing(self.store, [path])
        self._write("SOUL.md", "- Always answer in plain english.\n"
                               "- Never invent file paths you did not read.\n")
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
        self.assertEqual(import_standing(self.store, [path]), {"imported": 1, "files": 1})
        self.assertEqual(len(self.store.rules()), 2)

    def test_a_forced_rescan_does_not_duplicate_or_inflate(self):
        path = self._write("AGENTS.md", "- Always answer in plain english.\n")
        import_standing(self.store, [path])
        self.assertEqual(import_standing(self.store, [path], force=True),
                         {"imported": 0, "files": 1})
        rule = self.store.rules()[0]
        self.assertEqual(rule["hits"], 1)

    def test_an_import_that_repeats_a_stated_rule_adds_nothing(self):
        self.store.add("always answer in plain english and never hedge")
        path = self._write("SOUL.md", "- Always answer in plain english, never hedge your words.\n")
        self.assertEqual(import_standing(self.store, [path])["imported"], 0)
        rule = self.store.rules()[0]
        self.assertEqual(len(self.store.rules()), 1)
        self.assertEqual(rule["hits"], 1)
        self.assertEqual(rule["provenance"], "user_stated")

    def test_a_missing_file_is_skipped_quietly(self):
        self.assertEqual(import_standing(self.store, [self.root / "nope.md"]),
                         {"imported": 0, "files": 0})


if __name__ == "__main__":
    unittest.main()
