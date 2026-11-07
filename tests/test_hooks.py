"""Hook-level tests: the clearance rule and the gate, end to end through the plugin.

These exercise the actual ``pre_llm_call`` / ``transform_llm_output`` entry points with a
fake ctx, so the behaviour Hermes sees is the behaviour under test -- not an internal
function that happens to be called by them.
"""

from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
