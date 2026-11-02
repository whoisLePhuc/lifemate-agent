"""Canonical package boundaries and baseline contracts, without model loading."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


class StructureTests(unittest.TestCase):
    def test_old_packages_are_not_importable(self):
        for name in ['lifemate_agent','lifemate_data','lifemate_tools','lifemate_output','lifemate_evaluation']:
            with self.subTest(package=name):
                self.assertIsNone(importlib.util.find_spec(name))

    def test_baseline_prompt_and_contracts_are_unchanged(self):
        from lifemate.agent.prompt import SYSTEM
        from lifemate.contracts import Outcome, AgentInput
        from lifemate.input.adapter import AgentInput as InputType
        repo=Path(__file__).resolve().parents[2]
        summary=json.loads((repo/'Docs/Results/text-baseline-v0.1.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(SYSTEM.encode()).hexdigest(),summary['inference']['prompt_sha256'])
        from lifemate.agent.controller import Outcome as ControllerOutcome
        self.assertIs(Outcome,ControllerOutcome)
        self.assertIs(AgentInput,InputType)


if __name__=='__main__':unittest.main()
