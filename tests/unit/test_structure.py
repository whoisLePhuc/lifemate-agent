"""Refactor compatibility and shared contracts, without model loading."""
import hashlib
import json
from pathlib import Path
import unittest


class StructureTests(unittest.TestCase):
    def test_legacy_names_share_the_actual_classes(self):
        from lifemate_data import DatasetLoader as OldLoader
        from lifemate.input.adapter import DatasetLoader
        from lifemate_tools import ToolEnvAdapter as OldEnv
        from lifemate.tools.env_adapter import ToolEnvAdapter
        from lifemate_output import SubmissionWriter as OldWriter
        from lifemate.output.writer import SubmissionWriter
        from lifemate_evaluation.replay import TrainReplaySession as OldReplay
        from evaluation.replay import TrainReplaySession
        self.assertIs(OldLoader,DatasetLoader)
        self.assertIs(OldEnv,ToolEnvAdapter)
        self.assertIs(OldWriter,SubmissionWriter)
        self.assertIs(OldReplay,TrainReplaySession)

    def test_baseline_prompt_and_contracts_are_unchanged(self):
        from lifemate_agent.text import SYSTEM as OldPrompt, Outcome as OldOutcome
        from lifemate.agent.prompt import SYSTEM
        from lifemate.contracts import Outcome, AgentInput
        from lifemate.input.adapter import AgentInput as InputType
        repo=Path(__file__).resolve().parents[2]
        summary=json.loads((repo/'Docs/Results/text-baseline-v0.1.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(SYSTEM.encode()).hexdigest(),summary['inference']['prompt_sha256'])
        self.assertEqual(SYSTEM,OldPrompt)
        self.assertIs(Outcome,OldOutcome)
        self.assertIs(AgentInput,InputType)


if __name__=='__main__':unittest.main()
