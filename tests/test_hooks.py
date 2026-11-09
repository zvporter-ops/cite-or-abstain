"""Hook-level tests: the clearance rule and the gate, end to end through the plugin.

These exercise the actual ``pre_llm_call`` / ``transform_llm_output`` entry points with a
fake ctx, so the behaviour Hermes sees is the behaviour under test -- not an internal
function that happens to be called by them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def load_plugin():
    """Import the plugin package (its directory name is not a valid identifier)."""
    package = types.ModuleType("coa_plugin")
    package.__path__ = [str(ROOT)]
    sys.modules["coa_plugin"] = package
    spec = importlib.util.spec_from_file_location("coa_plugin", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["coa_plugin"] = module
    spec.loader.exec_module(module)
    return module


class FakeCtx:
    def __init__(self, settings):
        self.settings = settings

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def set_config(self, key, value):
        self.settings[key] = value


PLUGIN = load_plugin()


class HookTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.internal = root / "internal"
        self.public = root / "public"
        self.internal.mkdir()
        self.public.mkdir()
        (self.internal / "memo.md").write_text(
            "Project Falcon budget is 250000. The launch window is March 14.", encoding="utf-8")
        (self.public / "handbook.md").write_text(
            "A rollback window is the period during which a deployment can be reverted.",
            encoding="utf-8")

        self.home = root / "hermes"
        self.env = mock.patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        self.env.start()
        PLUGIN._CTX = FakeCtx({
            "corpus_roots": [{"path": str(self.internal), "tier": "internal"}],
            "public_roots": [str(self.public)],
            "strictness": "flag",
        })
        PLUGIN._INDEX = None
        PLUGIN._LAST.clear()
        PLUGIN._STRICTNESS.clear()

    def tearDown(self):
        self.env.stop()
        PLUGIN._CTX = None
        PLUGIN._INDEX = None
        PLUGIN._LAST.clear()
        PLUGIN._STRICTNESS.clear()
        self.tmp.cleanup()

    def pre(self, message: str, session: str = "s1", local: bool = True):
        with mock.patch.object(PLUGIN, "_session_is_local", return_value=local):
            return PLUGIN.on_pre_llm_call(session_id=session, user_message=message)

    def post(self, text: str, session: str = "s1"):
        return PLUGIN.on_transform_llm_output(response_text=text, session_id=session)


class TestClearance(HookTestCase):
    def test_internal_material_reaches_a_local_model(self):
        result = self.pre("What is the Project Falcon budget?")
        self.assertIn("EVIDENCE", result["context"])
        self.assertIn("250000", result["context"])
        self.assertIn("[internal]", result["context"])
        self.assertEqual(PLUGIN._LAST["s1"]["internal_withheld"], 0)

    def test_internal_material_is_withheld_from_a_remote_model(self):
        result = self.pre("What is the Project Falcon budget?", local=False)
        self.assertNotIn("250000", result["context"])
        self.assertNotIn("Falcon", result["context"])
        self.assertIn("NOT been included", result["context"])
        self.assertEqual(PLUGIN._LAST["s1"]["internal_withheld"], 1)
        self.assertFalse([p for p in PLUGIN._LAST["s1"]["passages"]
                          if p.tier == "internal"])

    def test_public_material_still_flows_to_a_remote_model(self):
        result = self.pre("What is a rollback window?", local=False)
        self.assertIn("rollback window", result["context"])
        self.assertIn("[public]", result["context"])
        self.assertEqual(PLUGIN._LAST["s1"]["internal_withheld"], 0)

    def test_silence_is_enforced_when_nothing_matches(self):
        result = self.pre("What is the soil mix for tomatoes?")
        self.assertIn("No internal or public material", result["context"])
        self.assertEqual(PLUGIN._LAST["s1"]["passages"], [])

    def test_non_questions_are_left_alone(self):
        self.assertIsNone(self.pre("thanks, that worked"))


class TestGateInTurn(HookTestCase):
    def test_strict_withholds_a_fabricated_figure(self):
        self.pre("What is the Project Falcon budget?")
        PLUGIN._STRICTNESS["s1"] = "strict"
        reply = self.post("The Project Falcon budget is 918,442 units.")
        self.assertIn("can't answer that from your corpus", reply)
        self.assertNotIn("918,442", reply)

    def test_strict_withholds_internal_when_the_model_is_remote(self):
        self.pre("What is the Project Falcon budget?", local=False)
        PLUGIN._STRICTNESS["s1"] = "strict"
        reply = self.post("The budget is 250000.")
        self.assertIn("withheld", reply)
        self.assertNotIn("250000", reply)

    def test_flag_appends_the_report_without_blocking(self):
        self.pre("What is the Project Falcon budget?")
        reply = self.post("The Project Falcon budget is 918,442 units.")
        self.assertIn("918,442", reply)
        self.assertIn("Evidence gate", reply)
        self.assertIn("unsupported claim", reply)

    def test_off_returns_the_reply_untouched(self):
        self.pre("What is the Project Falcon budget?")
        PLUGIN._STRICTNESS["s1"] = "off"
        self.assertIsNone(self.post("The Project Falcon budget is 918,442 units."))

    def test_supported_reply_is_not_annotated_when_nothing_is_flagged(self):
        self.pre("What is the Project Falcon budget?")
        reply = self.post("The Project Falcon budget is 250000.")
        self.assertIn("Evidence gate", reply)
        self.assertIn("No unsupported claims", reply)


class RecordingLLM:
    """Records every prompt and returns one response that parses in all three phases."""

    def __init__(self, answer: str = "A rollback window is the period during which a "
                                     "deployment can be reverted."):
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return ("CONFIDENCE: high\n"
                f"ANSWER: {self.answer}\n"
                "LIMITATIONS: none\n"
                "RECOMMENDATIONS: none")

    def text(self) -> str:
        return "\n".join(system + "\n" + user for system, user in self.calls)


class TestToolClearance(HookTestCase):
    """``corpus_ask`` and ``corpus_audit`` answer through the same model as the turn.

    With no ``endpoint`` configured they call Hermes' active model, which may be a cloud
    model. Internal passages must not reach it: the README promises a cloud model never
    sees private text, and the clearance notice points at ``corpus_ask`` as the local way
    out. If the tool forwards private text anyway, the promise is false.
    """

    def ask(self, llm, question="What is the Project Falcon budget for the rollback window?"):
        with mock.patch.object(PLUGIN, "_llm", return_value=llm):
            return json.loads(PLUGIN._tool_corpus_ask({"question": question}))

    def test_corpus_ask_withholds_internal_material_from_a_remote_session_model(self):
        PLUGIN._CTX.set_config("endpoint", "")
        llm = RecordingLLM()
        with mock.patch.object(PLUGIN, "_session_is_local", return_value=False):
            payload = self.ask(llm)
        self.assertNotIn("250000", llm.text())
        self.assertNotIn("250000", json.dumps(payload))
        self.assertEqual(payload["tier"], "public")
        self.assertEqual(payload["internal_withheld"], 1)
        self.assertIn("public material only", payload["notice"])

    def test_corpus_ask_withholds_internal_material_from_a_cloud_endpoint(self):
        PLUGIN._CTX.set_config("endpoint", "https://api.openai.com/v1")
        llm = RecordingLLM()
        with mock.patch.object(PLUGIN, "_session_is_local", return_value=True):
            payload = self.ask(llm)
        self.assertNotIn("250000", llm.text())
        self.assertNotEqual(payload["tier"], "internal")
        self.assertEqual(payload["internal_withheld"], 1)

    def test_corpus_ask_uses_internal_material_with_a_local_endpoint(self):
        PLUGIN._CTX.set_config("endpoint", "http://127.0.0.1:8888/v1")
        llm = RecordingLLM(answer="The Project Falcon budget is 250000.")
        with mock.patch.object(PLUGIN, "_session_is_local", return_value=False):
            payload = self.ask(llm)
        self.assertIn("250000", llm.text())
        self.assertEqual(payload["tier"], "internal")
        self.assertEqual(payload["internal_withheld"], 0)

    def test_corpus_ask_abstains_rather_than_leaking_when_only_internal_material_matches(self):
        PLUGIN._CTX.set_config("endpoint", "")
        PLUGIN._CTX.set_config("public_roots", [])
        PLUGIN._INDEX = None
        llm = RecordingLLM()
        with mock.patch.object(PLUGIN, "_session_is_local", return_value=False):
            payload = self.ask(llm)
        self.assertNotIn("250000", llm.text())
        self.assertTrue(payload["abstained"])
        self.assertEqual(payload["tier"], "none")

    def test_trust_refuses_when_internal_material_cannot_be_shared(self):
        PLUGIN._CTX.set_config("endpoint", "")
        with mock.patch.object(PLUGIN, "_session_is_local", return_value=False):
            report = PLUGIN._trust_report()
        self.assertIn("error", report)
        self.assertIn("local endpoint", report["error"])

    def test_trust_runs_when_a_local_endpoint_is_configured(self):
        PLUGIN._CTX.set_config("endpoint", "http://127.0.0.1:8888/v1")
        llm = RecordingLLM()
        with mock.patch.object(PLUGIN, "_session_is_local", return_value=False), \
                mock.patch.object(PLUGIN, "_llm", return_value=llm):
            report = PLUGIN._trust_report()
        self.assertNotIn("error", report)
        self.assertIn("verdict", report)
        self.assertIn("250000", llm.text())


class TestKeyHandling(HookTestCase):
    def test_api_key_is_read_only_through_the_secret_scope(self):
        """A stray ``COA_API_KEY`` in the environment must not become the bearer token."""
        PLUGIN._CTX.set_config("endpoint", "http://127.0.0.1:8888/v1")
        captured = {}

        def fake_endpoint(base, key="", model=""):
            captured["key"] = key
            return RecordingLLM()

        with mock.patch.dict(os.environ, {"COA_API_KEY": "from-env"}), \
                mock.patch.dict(sys.modules, {"agent.secret_scope": None}), \
                mock.patch.object(PLUGIN, "endpoint_llm", side_effect=fake_endpoint):
            PLUGIN._llm()
        self.assertEqual(captured["key"], "")


class TestClearanceNotice(HookTestCase):
    """The notice and the strict-mode reply must not advertise a path that cannot work."""

    def test_remote_notice_does_not_recommend_corpus_ask_without_a_local_endpoint(self):
        PLUGIN._CTX.set_config("endpoint", "")
        result = self.pre("What is the Project Falcon budget?", local=False)
        self.assertNotIn("answers it locally", result["context"])
        self.assertIn("local endpoint", result["context"])

    def test_remote_notice_recommends_corpus_ask_when_a_local_endpoint_exists(self):
        PLUGIN._CTX.set_config("endpoint", "http://127.0.0.1:8888/v1")
        result = self.pre("What is the Project Falcon budget?", local=False)
        self.assertIn("corpus_ask", result["context"])

    def test_strict_withheld_reply_does_not_point_at_a_path_that_cannot_work(self):
        PLUGIN._CTX.set_config("endpoint", "")
        self.pre("What is the Project Falcon budget?", local=False)
        PLUGIN._STRICTNESS["s1"] = "strict"
        reply = self.post("The budget is 250000.")
        self.assertNotIn("Use `corpus_ask`", reply)
        self.assertIn("local endpoint", reply)


class TestModeNormalization(HookTestCase):
    """``hermes config set ... strictness off`` stores a YAML boolean, not the string.

    If that arrives as False and is treated as unknown, the gate keeps annotating after
    the user asked for silence -- the plugin would appear not to obey its own setting.
    """

    def test_booleans_map_to_modes(self):
        self.assertEqual(PLUGIN._normalize_mode(False), "off")
        self.assertEqual(PLUGIN._normalize_mode(True), "flag")

    def test_ints_map_to_modes(self):
        # YAML can hand back 0/1 for a flag; 0 means off, anything else stays flag.
        self.assertEqual(PLUGIN._normalize_mode(0), "off")
        self.assertEqual(PLUGIN._normalize_mode(1), "flag")

    def test_strings_are_case_and_space_insensitive(self):
        self.assertEqual(PLUGIN._normalize_mode("STRICT"), "strict")
        self.assertEqual(PLUGIN._normalize_mode(" off "), "off")
        self.assertEqual(PLUGIN._normalize_mode("flag"), "flag")

    def test_unknown_and_missing_fall_back_to_flag(self):
        self.assertEqual(PLUGIN._normalize_mode("banana"), "flag")
        self.assertEqual(PLUGIN._normalize_mode(None), "flag")
        self.assertEqual(PLUGIN._normalize_mode(""), "flag")

    def test_boolean_false_in_config_silences_the_gate(self):
        self.pre("What is the Project Falcon budget?")
        PLUGIN._CTX.set_config("strictness", False)
        self.assertIsNone(self.post("The Project Falcon budget is 918,442 units."))

    def test_boolean_true_in_config_annotates(self):
        self.pre("What is the Project Falcon budget?")
        PLUGIN._CTX.set_config("strictness", True)
        reply = self.post("The Project Falcon budget is 918,442 units.")
        self.assertIn("Evidence gate", reply)

    def test_ground_reports_the_normalized_mode(self):
        PLUGIN._CTX.set_config("strictness", False)
        message = PLUGIN._cmd_ground("nonsense")
        self.assertIn("off", message)
        self.assertNotIn("False", message)


class TestCorrectionLedger(HookTestCase):
    """A correction becomes a rule; stating it again makes it hard and injected.

    This is the "never make me say it three times" layer: capture has to work on turns
    that are not questions (corrections rarely are), it has to survive the evidence gate
    being off, and repetition has to be detectable across turns.
    """

    def setUp(self):
        super().setUp()
        PLUGIN._RULES = None
        PLUGIN._LAST_MSG.clear()
        PLUGIN._TURN.clear()
        PLUGIN._TURN_DONE.clear()

    def tearDown(self):
        PLUGIN._RULES = None
        PLUGIN._LAST_MSG.clear()
        PLUGIN._TURN.clear()
        PLUGIN._TURN_DONE.clear()
        super().tearDown()

    CORRECTION = "no, use plain language and drop the hedging footer"

    def test_a_correction_is_captured_and_announced(self):
        self.pre(self.CORRECTION)
        reply = self.post("Understood.")
        self.assertIn("Rule 1 saved", reply)
        self.assertIn("plain language", reply)
        self.assertEqual(len(PLUGIN._rules_store().rules()), 1)

    def test_the_notice_is_its_own_block(self):
        self.pre(self.CORRECTION)
        reply = self.post("Understood.")
        self.assertIn("Understood.\n\n---\nRule 1 saved", reply)

    def test_capture_works_while_the_evidence_gate_is_off(self):
        PLUGIN._CTX.set_config("strictness", False)
        self.pre(self.CORRECTION)
        reply = self.post("Understood.")
        self.assertIn("Rule 1 saved", reply)

    def test_the_same_correction_twice_escalates_to_hard(self):
        self.pre(self.CORRECTION)
        self.post("Understood.")
        self.pre(self.CORRECTION)
        reply = self.post("Understood.")
        self.assertIn("hard", reply.lower())
        self.assertIn("stated before", reply.lower())
        self.assertTrue(PLUGIN._rules_store().rules()[0]["hard"])

    def test_a_hard_rule_is_injected_into_the_next_turn(self):
        self.pre(self.CORRECTION)
        self.post("Understood.")
        self.pre(self.CORRECTION)
        self.post("Understood.")
        result = self.pre("What is the Project Falcon budget?")
        self.assertIn("STANDING RULES", result["context"])
        self.assertIn("plain language", result["context"])
        self.assertIn("EVIDENCE", result["context"])

    def test_rules_are_injected_even_when_the_corpus_is_empty(self):
        PLUGIN._CTX.set_config("corpus_roots", [])
        PLUGIN._CTX.set_config("public_roots", [])
        PLUGIN._INDEX = None
        self.pre(self.CORRECTION)
        self.post("Understood.")
        result = self.pre("thanks")
        self.assertIn("STANDING RULES", result["context"])

    def test_a_question_is_not_captured(self):
        self.pre("What is the Project Falcon budget?")
        reply = self.post("The Project Falcon budget is 250000.")
        self.assertNotIn("Rule 1", reply)
        self.assertIn("Evidence gate", reply)
        self.assertEqual(PLUGIN._rules_store().rules(), [])

    def test_an_ordinary_statement_is_not_captured(self):
        self.pre("The Project Falcon budget is 250000.")
        reply = self.post("Noted.")
        self.assertIsNone(reply)
        self.assertEqual(PLUGIN._rules_store().rules(), [])

    def test_one_turn_is_counted_once(self):
        self.pre(self.CORRECTION)
        self.post("Understood.")
        self.post("Understood again.")
        rules = PLUGIN._rules_store().rules()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["hits"], 1)

    def test_capture_can_be_turned_off(self):
        PLUGIN._CTX.set_config("rules_enabled", False)
        self.pre(self.CORRECTION)
        self.assertIsNone(self.post("Understood."))
        self.assertEqual(PLUGIN._rules_store().rules(), [])

    def test_rules_survive_a_fresh_store(self):
        self.pre(self.CORRECTION)
        self.post("Understood.")
        PLUGIN._RULES = None
        result = self.pre("What is the Project Falcon budget?")
        self.assertIn("plain language", result["context"])

    def test_the_rules_command_lists_and_forgets(self):
        self.pre(self.CORRECTION)
        self.post("Understood.")
        listing = PLUGIN._cmd_rules("")
        self.assertIn("plain language", listing)
        self.assertIn("Rule 1 forgotten", PLUGIN._cmd_rules("forget 1"))
        self.assertEqual(PLUGIN._rules_store().rules(), [])

    def test_the_rules_command_reports_an_empty_store(self):
        self.assertIn("No standing rules yet", PLUGIN._cmd_rules(""))

    def test_the_rules_command_promotes(self):
        self.pre(self.CORRECTION)
        self.post("Understood.")
        self.assertIn("hard", PLUGIN._cmd_rules("hard 1").lower())
        self.assertTrue(PLUGIN._rules_store().rules()[0]["hard"])


class TestStandingImport(HookTestCase):
    """A cold start is not an empty start: rules the user already wrote are read in first."""

    SOUL = ("# Conventions\n\n"
            "- Always answer in plain english, no filler.\n"
            "- Never invent file paths you did not read.\n")

    def setUp(self):
        super().setUp()
        PLUGIN._RULES = None
        PLUGIN._LAST_MSG.clear()
        PLUGIN._TURN.clear()
        PLUGIN._TURN_DONE.clear()
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "SOUL.md").write_text(self.SOUL, encoding="utf-8")

    def tearDown(self):
        PLUGIN._RULES = None
        PLUGIN._LAST_MSG.clear()
        PLUGIN._TURN.clear()
        PLUGIN._TURN_DONE.clear()
        super().tearDown()

    def test_standing_files_seed_the_very_first_turn(self):
        result = self.pre("thanks, that worked")  # a non-question still gets the rules
        self.assertIn("STANDING RULES", result["context"])
        self.assertIn("[imported]", result["context"])
        self.assertIn("plain english", result["context"])
        self.assertEqual(len(PLUGIN._rules_store().rules()), 2)

    def test_import_can_be_turned_off(self):
        PLUGIN._CTX.set_config("rules_import", False)
        self.assertIsNone(self.pre("thanks, that worked"))
        self.assertEqual(PLUGIN._rules_store().rules(), [])

    def test_the_import_command_picks_up_a_new_bullet(self):
        self.pre("thanks, that worked")
        (self.home / "SOUL.md").write_text(
            self.SOUL + "- Use the vault copy instead of re-fetching.\n", encoding="utf-8")
        self.assertIn("1 new rule", PLUGIN._cmd_rules("import"))
        self.assertEqual(len(PLUGIN._rules_store().rules()), 3)

    def test_the_user_restating_an_imported_rule_hardens_and_relabels_it(self):
        self.pre("no, always answer in plain english and keep it short")
        self.post("Understood.")
        rule = [r for r in PLUGIN._rules_store().rules()
                if "plain english" in r["text"]][0]
        self.assertEqual(rule["provenance"], "user_stated")
        self.assertTrue(rule["hard"])
        self.assertIn("keep it short", rule["text"])

    def test_the_listing_marks_imported_rules(self):
        self.pre("thanks, that worked")
        self.assertIn("[imported]", PLUGIN._cmd_rules(""))


if __name__ == "__main__":
    unittest.main()
