"""Synthetic contract tests; never open the real environment directories."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lifemate.input.adapter import DataError, DatasetLoader


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tool = {"name": "notes.create", "app": "notes", "kind": "write",
                     "parameters": {"type": "object", "required": ["title"],
                                    "properties": {"title": {"type": "string", "x-match": "fuzzy"}}}}
        self.catalog = {"schema_version": "test", "apps": [
            {"id": "notes", "tools": [self.tool]},
            {"id": "sys", "tools": [{"name": "sys.ask_missing_info", "app": "sys",
                                        "parameters": {"type": "object"}}]}]}
        self.row = {"id": "synthetic", "audio": "audio/latest.wav", "audio_text": "GOLD_NOW",
                    "context": {"current_time": "2026-11-02T10:00:00+07:00",
                                "timezone": "Asia/Ho_Chi_Minh", "location": {},
                                "user_profile": {}, "installed_apps": {"notes": {"tools": ["notes.create"]}}},
                    "history": [{"role": "user", "audio": "audio/prior.wav", "audio_text": "GOLD_HISTORY"},
                                {"role": "assistant", "text": "Đã ghi nhận", "tool_calls": [], "tool_results": []}],
                    "label": {"tool_calls": [{"name": "notes.create", "arguments": {"title": "TARGET"}}],
                              "tool_results": [{"ok": True}], "final_state": {"notes": {}}}}

    def write(self, rows=None, split="train"):
        folder = self.root / split
        (folder / "audio").mkdir(parents=True, exist_ok=True)
        for name in ["latest.wav", "prior.wav"]:
            (folder / "audio" / name).touch()
        (folder / f"{split}_tools_catalog.json").write_text(json.dumps(self.catalog), encoding="utf-8")
        (folder / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in (rows if rows is not None else [self.row])) + "\n",
            encoding="utf-8")
        return folder

    def test_agent_input_excludes_gold_but_preserves_history(self):
        self.write()
        item = next(DatasetLoader(self.root, "train").iter_inputs())
        serialized = str(asdict(item))
        for forbidden in ["GOLD_NOW", "GOLD_HISTORY", "TARGET", "audio_text", "final_state", "label"]:
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(item.history[1]["text"], "Đã ghi nhận")
        self.assertTrue(item.audio.is_absolute())
        self.assertEqual([x["name"] for x in item.tools], ["notes.create", "sys.ask_missing_info"])
        self.assertEqual(item.tools[0]["parameters"], self.tool["parameters"])

    def test_reference_explicit_and_train_only(self):
        self.write()
        example = next(DatasetLoader(self.root, "train").iter_training())
        self.assertEqual(example.reference.audio_text, "GOLD_NOW")
        self.assertEqual(example.reference.history_audio_text, ("GOLD_HISTORY",))
        for split in ["public_test", "private_test"]:
            self.write(split=split)
            with self.assertRaisesRegex(DataError, "only for train"):
                next(DatasetLoader(self.root, split).iter_training())

    def test_schema_copies_are_isolated(self):
        self.write()
        loader = DatasetLoader(self.root, "train")
        first = next(loader.iter_inputs())
        first.tools[0]["parameters"].clear()
        self.assertEqual(next(loader.iter_inputs()).tools[0]["parameters"], self.tool["parameters"])

    def test_forbidden_paths(self):
        for value in ["../train_environment/x/state.json", "audio/../../state.json",
                      "audio/x_environment/state.json", "C:\\secret.wav", "/tmp/a.wav", "\\\\host\\file.wav"]:
            with self.subTest(value=value):
                row = deepcopy(self.row); row["audio"] = value
                self.write([row])
                with self.assertRaises(DataError):
                    next(DatasetLoader(self.root, "train", check_audio=False).iter_inputs())

    def test_invalid_allowlist(self):
        for installed in [{"notes": {"tools": ["notes.unknown"]}},
                          {"notes": {"tools": ["sys.ask_missing_info"]}},
                          {"unknown": {"tools": []}}]:
            row = deepcopy(self.row); row["context"]["installed_apps"] = installed
            self.write([row])
            with self.assertRaises(DataError):
                next(DatasetLoader(self.root, "train").iter_inputs())

    def test_duplicate_id_and_json_keys(self):
        folder = self.write([self.row, self.row])
        with self.assertRaises(DataError):
            list(DatasetLoader(self.root, "train").iter_inputs())
        (folder / "train.jsonl").write_text('{"id":"x","id":"y"}\n', encoding="utf-8")
        with self.assertRaises(DataError):
            list(DatasetLoader(self.root, "train").iter_inputs())

    def test_missing_audio_and_optional_metadata_mode(self):
        folder = self.write()
        (folder / "audio/latest.wav").unlink()
        with self.assertRaisesRegex(DataError, "missing audio"):
            next(DatasetLoader(self.root, "train").iter_inputs())
        self.assertEqual(next(DatasetLoader(self.root, "train", check_audio=False).iter_inputs()).id, "synthetic")

    def test_malformed_history_and_reference(self):
        row = deepcopy(self.row); row["history"][0]["role"] = "system"
        self.write([row])
        with self.assertRaisesRegex(DataError, "unsupported history"):
            next(DatasetLoader(self.root, "train").iter_inputs())
        row = deepcopy(self.row); row["label"]["tool_results"] = []
        self.write([row])
        with self.assertRaisesRegex(DataError, "lengths differ"):
            next(DatasetLoader(self.root, "train").iter_training())

    def test_reader_opens_only_jsonl_and_catalog(self):
        self.write()
        opened = []
        original = Path.open
        def guarded(path, *args, **kwargs):
            opened.append(path)
            if any(p.endswith("_environment") for p in path.parts):
                raise AssertionError("forbidden environment access")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            list(DatasetLoader(self.root, "train").iter_inputs())
        self.assertEqual({p.name for p in opened}, {"train.jsonl", "train_tools_catalog.json"})

    def test_line_error_does_not_expose_data(self):
        folder = self.write()
        (folder / "train.jsonl").write_text('{"secret":"PRIVATE_VALUE"\n', encoding="utf-8")
        with self.assertRaises(DataError) as caught:
            list(DatasetLoader(self.root, "train").iter_inputs())
        self.assertIn("line 1", str(caught.exception))
        self.assertNotIn("PRIVATE_VALUE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
