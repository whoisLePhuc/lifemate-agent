"""Checkpoint format, resume verification and CLI surface of the epoch trainer (CPU, no model)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from training import train_epoch as t


class TinyAdapter(torch.nn.Module):
    """Stands in for a PEFT model: same save_pretrained(path, safe_serialization=...) contract."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(3))

    def save_pretrained(self, path, safe_serialization=True):
        from safetensors.torch import save_file
        Path(path).mkdir(parents=True)
        save_file({'w': self.w.detach().contiguous()}, str(Path(path) / 'adapter_model.safetensors'))


CONFIG = {'order': [3, 1, 0, 2], 'accumulation': 2, 'optimizer_steps': 2, 'warmup_steps': 1, 'learning_rate': 1e-4}


def make_checkpoint(root, step=1, config=CONFIG):
    model = TinyAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return t.save_checkpoint(root, step, model, optimizer, config, {'token_mean_loss': 1.0}, 1.5, [{'step': step}])


class CheckpointTest(unittest.TestCase):
    def test_saved_checkpoint_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_checkpoint(Path(d))
            self.assertEqual(path.name, 'checkpoint-0001')
            self.assertEqual(t.verify_checkpoint(path, CONFIG)['step'], 1)

    def test_never_overwrites_an_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            make_checkpoint(Path(d))
            with self.assertRaises(ValueError):
                make_checkpoint(Path(d))

    def test_tampered_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_checkpoint(Path(d))
            with (path / 'adapter/adapter_model.safetensors').open('ab') as f:
                f.write(b'x')
            with self.assertRaises(ValueError):
                t.verify_checkpoint(path, CONFIG)

    def test_changed_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_checkpoint(Path(d))
            with self.assertRaises(ValueError):
                t.verify_checkpoint(path, {**CONFIG, 'learning_rate': 2e-4})

    def test_incomplete_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_checkpoint(Path(d))
            marker = json.loads((path / 'checkpoint.json').read_text())
            (path / 'checkpoint.json').write_text(json.dumps({**marker, 'status': 'partial'}))
            with self.assertRaises(ValueError):
                t.verify_checkpoint(path, CONFIG)

    def test_checkpoint_under_models_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_checkpoint(Path(d))
            fake = Path(d) / 'Models' / 'x'
            fake.parent.mkdir()
            path.rename(fake)
            with self.assertRaises(ValueError):
                t.verify_checkpoint(fake, CONFIG)


class StateHelpersTest(unittest.TestCase):
    def test_rng_state_round_trips(self):
        state = t.rng_state()
        a = torch.rand(3)
        t.restore_rng(state)
        self.assertTrue(torch.equal(a, torch.rand(3)))

    def test_tensor_tree_equal_detects_differences(self):
        a = {'x': [torch.ones(2), 1], 'y': {'z': torch.zeros(1)}}
        b = {'x': [torch.ones(2), 1], 'y': {'z': torch.zeros(1)}}
        self.assertTrue(t.tensor_tree_equal(a, b))
        b['y']['z'] = torch.ones(1)
        self.assertFalse(t.tensor_tree_equal(a, b))
        self.assertFalse(t.tensor_tree_equal({'x': 1}, {'x': 1, 'y': 2}))

    def test_sources_cover_only_the_training_package_with_relative_paths(self):
        names = set(t.sources())
        self.assertTrue({'training/train_epoch.py', 'training/qwen35.py', 'training/epoch_schedule.py'} <= names)
        self.assertTrue(all(n.startswith('training/') and not n.startswith('/') for n in names))


class CliTest(unittest.TestCase):
    def run_main(self, module, argv):
        with mock.patch.object(sys, 'argv', ['prog', *argv]), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                module.main()
        return caught.exception.code

    def test_dataset_and_model_paths_are_required(self):
        self.assertEqual(self.run_main(t, ['--output-dir', 'x']), 2)

    def test_help_exits_cleanly_for_both_commands(self):
        from training import verify_resume
        self.assertEqual(self.run_main(t, ['--help']), 0)
        self.assertEqual(self.run_main(verify_resume, ['--help']), 0)

    def test_unified_cli_registers_the_new_commands(self):
        from lifemate.cli import COMMANDS
        self.assertEqual(COMMANDS['train-epoch'], 'training.train_epoch')
        self.assertEqual(COMMANDS['verify-resume'], 'training.verify_resume')

    def test_output_inside_dataset_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ds = Path(d) / 'dataset'
            ds.mkdir()
            argv = ['--dataset-dir', str(ds), '--model-path', str(Path(d) / 'm'), '--output-dir', str(ds / 'out')]
            with mock.patch.object(sys, 'argv', ['prog', *argv]), mock.patch.object(t, 'prepare_data') as prepare:
                with self.assertRaises(ValueError):
                    t.main()
                prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
