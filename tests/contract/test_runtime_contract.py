"""Opt-in official-runtime contract test, train only, one ToolEnv in a fresh process."""
import os
import unittest

from lifemate.input.adapter import DatasetLoader
from lifemate.tools.env_adapter import LifecycleError, ToolEnvAdapter


@unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "set LIFEMATE_DATA_ROOT to opt in")
class RuntimeContractTests(unittest.TestCase):
    def test_official_rejection_ledger_and_validation(self):
        root = os.environ["LIFEMATE_DATA_ROOT"]
        item = next(DatasetLoader(root, "train").iter_inputs())
        with ToolEnvAdapter(root, "train", item.id) as adapter:
            missing_tool = "__adapter_nonexistent_tool__"
            self.assertFalse(adapter.validate(missing_tool, {})["ok"])
            self.assertEqual(adapter.get_tool_calls(), [])
            self.assertFalse(adapter.call(missing_tool, {})["ok"])
            record = adapter.finalize()
            self.assertEqual(record["tool_calls"], [{"name": missing_tool, "arguments": {}}])
            self.assertRegex(record["state_hash"], r"^[0-9a-f]{64}$")
            with self.assertRaises(LifecycleError):
                adapter.call(missing_tool, {})


if __name__ == "__main__": unittest.main()
