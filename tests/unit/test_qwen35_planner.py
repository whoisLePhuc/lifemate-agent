"""Qwen3.5 planner wrapper without loading a model."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lifemate.agent import planner as base_planner
from lifemate.agent import qwen35_planner as q


class FakeTokenizer:
    unk_token_id = 0
    ids = {'<|im_end|>': 11, '<|endoftext|>': 12}

    def __init__(self, ids=None):
        self.ids = ids or self.ids
        self.rendered = []

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token)

    def apply_chat_template(self, *args, **kwargs):
        self.rendered.append(kwargs)
        return 'ok'


def fake_init(self, model_path, **kwargs):
    self.tokenizer = FakeTokenizer(kwargs.pop('ids', None))
    self.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None))
    self.parameters = 4_200_000_000


class Qwen35PlannerTest(unittest.TestCase):
    def build(self, total=4_600_000_000, **kwargs):
        audit = {'checkpoint_total': total, 'checkpoint_vision': 1, 'checkpoint_non_vision': total - 1}
        with mock.patch.object(base_planner.LocalPlanner, '__init__', fake_init), \
                mock.patch.object(q, 'checkpoint_parameters', return_value=audit):
            return q.Qwen35Planner('unused', **kwargs)

    def test_thinking_disabled_on_every_render_and_stop_tokens_set(self):
        planner = self.build()
        planner.tokenizer.apply_chat_template([], tokenize=False)
        self.assertIs(planner.tokenizer.rendered[0]['enable_thinking'], False)
        self.assertEqual(planner.model.generation_config.eos_token_id, [11, 12])
        self.assertEqual(planner.parameter_audit['loaded_language_model'], 4_200_000_000)

    def test_missing_stop_token_rejected(self):
        with self.assertRaises(ValueError):
            self.build(ids={'<|im_end|>': 11})

    def test_checkpoint_at_or_over_limit_rejected(self):
        with self.assertRaises(ValueError):
            self.build(total=5_000_000_000)

    def test_checkpoint_parameter_count_separates_vision_tower(self):
        from safetensors.numpy import save_file
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_file({'model.language_model.w': np.zeros((2, 3), 'float32'),
                       'model.visual.w': np.zeros((4,), 'float32')}, str(d / 's.safetensors'))
            (d / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'model.language_model.w': 's.safetensors', 'model.visual.w': 's.safetensors'}}))
            self.assertEqual(q.checkpoint_parameters(d),
                             {'checkpoint_total': 10, 'checkpoint_vision': 4, 'checkpoint_non_vision': 6})


if __name__ == '__main__':
    unittest.main()
