"""predict orchestration: command order and arguments, without running any model."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lifemate.pipeline import predict as p


def run_main(argv):
    with mock.patch.object(sys, 'argv', ['lifemate predict', *argv]), \
            mock.patch.object(p.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
        p.main()
    return run


class PredictTest(unittest.TestCase):
    def args(self, d, **extra):
        base = ['--split', 'public_test', '--data-root', d, '--asr-model', 'asr', '--llm-model', 'llm',
                '--adapter', 'ad', '--asr-python', '/asr/python', '--llm-python', '/llm/python',
                '--output-dir', str(Path(d) / 'run')]
        for k, v in extra.items():
            base += [f'--{k.replace("_", "-")}', v]
        return base

    def test_three_stages_in_order_with_their_own_interpreters(self):
        with tempfile.TemporaryDirectory() as d:
            run = run_main(self.args(d))
            commands = [c.args[0] for c in run.call_args_list]
            self.assertEqual(len(commands), 3)
            self.assertEqual(commands[0][:4], ['/asr/python', '-m', 'lifemate.cli', 'transcribe'])
            self.assertEqual(commands[1][:5], ['/llm/python', '-u', '-m', 'lifemate.cli', 'run-agent'])
            self.assertEqual(commands[2][1:3], ['-m', 'zaic_tools'])
            out = str(Path(d).resolve() / 'run')
            self.assertIn(f'{out}/asr/transcripts.jsonl', commands[1])
            self.assertEqual(commands[2][-1], f'{out}/agent/inference/submission.jsonl')
            self.assertEqual(run.call_args_list[2].kwargs['cwd'], Path(d).resolve() / 'public_test')
            self.assertTrue(all(c.kwargs['check'] for c in run.call_args_list))

    def test_ids_file_and_deadline_reach_both_stages(self):
        with tempfile.TemporaryDirectory() as d:
            run = run_main(self.args(d, ids_file='ids.json', seconds='20'))
            transcribe, agent = (c.args[0] for c in run.call_args_list[:2])
            self.assertEqual(transcribe[transcribe.index('--ids-file') + 1], 'ids.json')
            self.assertEqual(agent[agent.index('--ids-file') + 1], 'ids.json')
            self.assertEqual(agent[agent.index('--seconds') + 1], '20.0')

    def test_offline_environment_is_forced(self):
        with tempfile.TemporaryDirectory() as d:
            env = run_main(self.args(d)).call_args_list[0].kwargs['env']
            self.assertEqual((env['HF_HUB_OFFLINE'], env['TRANSFORMERS_OFFLINE']), ('1', '1'))

    def test_a_failing_stage_stops_the_pipeline(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sys, 'argv', ['x', *self.args(d)]), \
                mock.patch.object(p.subprocess, 'run', side_effect=p.subprocess.CalledProcessError(2, 'x')) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(p.subprocess.CalledProcessError):
                p.main()
            self.assertEqual(run.call_count, 1)

    def test_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'run').mkdir()
            with mock.patch.object(sys, 'argv', ['x', *self.args(d)]), mock.patch.object(p.subprocess, 'run') as run:
                with self.assertRaises(SystemExit):
                    p.main()
                run.assert_not_called()

    def test_test_splits_only(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sys, 'argv', ['x', *self.args(d)[:1], 'train', *self.args(d)[2:]]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                p.main()


if __name__ == '__main__':
    unittest.main()
