from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lifemate.output.writer import SubmissionError, SubmissionWriter, validate_submission
from evaluation.replay import TrainReplaySession


class WriterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "submission.jsonl"
        self.record = {"id": "a", "tool_calls": [{"name": "sms.send", "arguments": {"content": "Tiếng Việt\nxin chào"}}],
                       "state_hash": "a" * 64}
    def test_roundtrip_and_preserve_calls(self):
        with SubmissionWriter(self.path, ["a", "b"]) as writer:
            writer.write(self.record)
            writer.write({"id": "b", "tool_calls": [], "state_hash": "b" * 64})
            self.assertFalse(self.path.exists())
        self.assertEqual(validate_submission(self.path, ["a", "b"])["records"], 2)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), self.record)
    def test_missing_id_no_final_file(self):
        with self.assertRaises(SubmissionError):
            with SubmissionWriter(self.path, ["a", "b"]) as writer:
                writer.write(self.record)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [])
    def test_duplicate_and_unknown_id(self):
        for record in [self.record, {**self.record, "id": "unknown"}]:
            with self.subTest(id=record["id"]):
                with self.assertRaises(SubmissionError):
                    with SubmissionWriter(self.path, ["a"]) as writer:
                        writer.write(self.record); writer.write(record)
                self.assertFalse(self.path.exists())
    def test_invalid_structure_hash_and_json(self):
        bad = [{**self.record, "extra": True}, {**self.record, "state_hash": "fake"},
               {**self.record, "tool_calls": {}}, {**self.record, "tool_calls": [{"name": "x", "arguments": {}, "result": {}}]},
               {**self.record, "tool_calls": [{"name": "x", "arguments": {"x": float("nan")}}]}]
        for record in bad:
            with self.assertRaises(SubmissionError):
                with SubmissionWriter(self.path, ["a"]) as writer: writer.write(record)
        with self.assertRaises(SubmissionError): SubmissionWriter(self.path, ["a", "a"])
    def test_never_overwrite_even_publish_race(self):
        with self.assertRaises(FileExistsError):
            with SubmissionWriter(self.path, ["a"]) as writer:
                writer.write(self.record)
                self.path.write_text("existing", encoding="utf-8")
        self.assertEqual(self.path.read_text(), "existing")
        with self.assertRaises(FileExistsError): SubmissionWriter(self.path, ["a"])
    def test_exception_or_disk_failure_aborts(self):
        with self.assertRaises(RuntimeError):
            with SubmissionWriter(self.path, ["a"]) as writer: raise RuntimeError("stopped")
        self.assertFalse(self.path.exists())
        with patch("lifemate.output.writer.os.fsync", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                with SubmissionWriter(self.path, ["a"]) as writer: writer.write(self.record)
        self.assertFalse(self.path.exists())
    def test_readback_detects_corruption(self):
        for text in ['\n', '{"id":"a","id":"a"}\n', json.dumps(self.record)+"\n"+json.dumps(self.record)+"\n"]:
            self.path.write_text(text, encoding="utf-8")
            with self.assertRaises(SubmissionError): validate_submission(self.path, ["a"])
        self.path.write_text(json.dumps(self.record)+"\n", encoding="utf-8")
        with self.assertRaises(SubmissionError): validate_submission(self.path, ["a", "b"])


class ReplayTests(unittest.TestCase):
    def test_changed_and_unchanged_apps_checked_without_exposing_state(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        class Fake:
            def __init__(self, *args, **kwargs): self.state = {"notes": {"x": 1}, "sms": {"x": 2}}
            def get_state(self): return deepcopy(self.state)
        with TrainReplaySession(temp.name, "train_fake", toolenv_factory=Fake) as env:
            env._session.state["notes"] = {"x": 3}
            check = env.compare_final_state({"notes": {"x": 3}})
            self.assertTrue(check["full_state_match"])
            self.assertEqual(check["label_apps_matched"], 1)
            env._session.state["sms"] = {"x": 999}
            self.assertFalse(env.compare_final_state({"notes": {"x": 3}})["full_state_match"])
            self.assertNotIn("state", check)


if __name__ == "__main__": unittest.main()
