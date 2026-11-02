from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from lifemate.tools.env_adapter import AdapterError, LifecycleError, RuntimeFailure, ToolEnvAdapter


class FakeEnv:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.hash_reads = 0
        self.explode = False
    def validate(self, name, arguments):
        return {"ok": name == "demo.write", "error": None if name == "demo.write" else "schema"}
    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": deepcopy(arguments)})
        if self.explode:
            raise TimeoutError("unknown side effect")
        return {"ok": False, "error": "schema"} if name == "unknown" else {"ok": True, "done": False}
    def get_tool_calls(self):
        return self.calls
    def get_state_hash(self):
        self.hash_reads += 1
        return "runtime_hash"


class AdapterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.envs = []
        def factory(*args, **kwargs):
            env = FakeEnv(*args, **kwargs)
            self.envs.append(env)
            return env
        self.factory = factory
    def adapter(self, id="sample"):
        return ToolEnvAdapter(self.root, "train", id, toolenv_factory=self.factory)
    def test_one_instance_and_no_reset_after_close(self):
        adapter = self.adapter()
        with self.assertRaises(LifecycleError): self.adapter()
        adapter.close()
        with self.assertRaises(LifecycleError): self.adapter()
        self.assertEqual(len(self.envs), 1)
    def test_validate_does_not_record_and_call_error_does(self):
        a = self.adapter()
        self.assertFalse(a.validate("unknown", {})["ok"])
        self.assertEqual(a.get_tool_calls(), [])
        self.assertFalse(a.call("unknown", {})["ok"])
        self.assertEqual(a.finalize()["tool_calls"], [{"name": "unknown", "arguments": {}}])
    def test_finalize_cached_copied_and_terminal(self):
        a = self.adapter()
        args = {"nested": [1]}
        result = a.call("demo.write", args)
        self.assertTrue(result["ok"])
        self.assertFalse(result["done"])
        args["nested"].append(2)
        record = a.finalize()
        record["tool_calls"].clear()
        self.assertEqual(a.finalize()["tool_calls"][0]["arguments"], {"nested": [1]})
        self.assertEqual(self.envs[0].hash_reads, 1)
        with self.assertRaises(LifecycleError): a.call("demo.write", {})
        with self.assertRaises(LifecycleError): a.validate("demo.write", {})
        a.close()
        self.assertEqual(a.finalize()["state_hash"], "runtime_hash")
    def test_unexpected_failure_poisoned_no_retry(self):
        a = self.adapter(); self.envs[0].explode = True
        with self.assertRaises(RuntimeFailure): a.call("demo.write", {})
        self.assertEqual(a.status, "broken")
        with self.assertRaises(LifecycleError): a.call("demo.write", {})
        with self.assertRaises(LifecycleError): a.finalize()
        self.assertEqual(len(self.envs[0].calls), 1)
    def test_failed_construction_cannot_retry(self):
        def fail(*args, **kwargs): raise OSError("missing")
        with self.assertRaises(RuntimeFailure):
            ToolEnvAdapter(self.root, "train", "sample", toolenv_factory=fail)
        with self.assertRaises(LifecycleError): self.adapter()
    def test_invalid_inputs_never_dispatch(self):
        a = self.adapter()
        for args in [[], {"x": float("nan")}, {1: "x"}, {"x": object()}]:
            with self.subTest(args=type(args).__name__):
                with self.assertRaises(AdapterError): a.call("demo.write", args)
        self.assertEqual(a.get_tool_calls(), [])
        for sample_id in ["../x", "C:\\x", "a/b", "", "a.b"]:
            with self.assertRaises(AdapterError): self.adapter(sample_id)
    def test_context_manager_does_not_auto_finalize(self):
        with self.adapter() as a:
            a.call("demo.write", {})
        self.assertEqual(a.status, "closed")
        with self.assertRaises(LifecycleError): a.finalize()
        self.assertEqual(self.envs[0].hash_reads, 0)
    def test_empty_record_and_independent_samples(self):
        a, b = self.adapter("a"), self.adapter("b")
        a.call("demo.write", {})
        self.assertEqual(b.finalize(), {"id": "b", "tool_calls": [], "state_hash": "runtime_hash"})
    def test_invalid_hash_no_fallback(self):
        a = self.adapter()
        self.envs[0].get_state_hash = lambda: ""
        with self.assertRaises(RuntimeFailure): a.finalize()
        self.assertEqual(a.status, "broken")


if __name__ == "__main__": unittest.main()
