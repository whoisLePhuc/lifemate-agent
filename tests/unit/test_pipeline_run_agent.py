"""LLM stage: transcript handling and the production loop, with a fake planner and ToolEnv (CPU)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lifemate.contracts import AgentInput
from lifemate.pipeline import run_agent as r
from lifemate.tools.env_adapter import ToolEnvAdapter

HASH = 'ab' * 32  # the submission writer requires a lowercase SHA-256 hex string
TOOLS = [{'name': 'demo.write', 'kind': 'write', 'description': 'd', 'parameters': {}},
         {'name': 'sys.ask_missing_info', 'kind': 'meta', 'description': 'ask', 'parameters': {}}]


class FakeEnv:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def validate(self, name, arguments):
        return {'ok': 'value' in arguments, 'error': 'required value'}

    def call(self, name, arguments):
        self.calls.append({'name': name, 'arguments': arguments})
        return {'ok': True, 'done': True}

    def get_tool_calls(self):
        return self.calls

    def get_state_hash(self):
        return HASH


class ScriptedPlanner:
    def __init__(self, scripts):
        self.scripts = scripts
        self.seen = []
        self.parameters = 1

    def generate(self, messages, seconds, **kwargs):
        self.seen.append(json.dumps(messages, ensure_ascii=False))
        output = self.scripts.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def item(sid):
    return AgentInput(sid, Path(f'{sid}.wav'), {'current_time': 'x'}, [], TOOLS)


def write(path, rows):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')


class TranscriptFileTest(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as d:
            write(Path(d) / 't.jsonl', rows)
            return r.load_transcripts(Path(d) / 't.jsonl')

    def test_valid_file(self):
        self.assertIn('s1', self.load([{'id': 's1', 'request': 'r', 'history': ['h']}]))

    def test_rejects_extra_keys_duplicates_and_wrong_types(self):
        good = {'id': 's1', 'request': 'r', 'history': []}
        for bad in ([{**good, 'gold': 'x'}], [good, good], [{**good, 'request': None}], [{**good, 'history': [1]}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.load(bad)


class AsrInputTest(unittest.TestCase):
    def test_gold_strings_are_discarded_and_forwarded_text_is_journaled(self):
        rows = []
        seen = {}

        def original(it, request, history):
            seen.update(request=request, history=history)
            return {'request': request, 'history': [{'role': 'user', 'text': t} for t in history]}
        build = r.asr_text_input(original, {'s1': {'id': 's1', 'request': 'asr', 'history': ['h']}}, rows.append)
        build(SimpleNamespace(id='s1'), 'GOLD', ('GOLD H',))
        self.assertEqual(seen, {'request': 'asr', 'history': ('h',)})
        self.assertEqual(rows[0], {'id': 's1', 'request_sha256': r.text_sha('asr'), 'history_sha256': [r.text_sha('h')]})


class MainTest(unittest.TestCase):
    def argv(self, out):
        return ['run-agent', '--split', 'public_test', '--data-root', 'd', '--model-path', 'm', '--adapter-path', 'a',
                '--transcripts', 't.jsonl', '--output-dir', str(out)]

    def test_a_failed_model_load_leaves_no_output_directory(self):
        import sys
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            with mock.patch.object(sys, 'argv', self.argv(out)), \
                    mock.patch.object(r, 'load_planner', side_effect=RuntimeError('no model')):
                with self.assertRaises(RuntimeError):
                    r.main()
            self.assertFalse(out.exists())

    def test_oracle_is_refused_on_a_test_split_before_anything_loads(self):
        import sys
        from unittest import mock
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(sys, 'argv', [*self.argv(Path(d) / 'run'), '--mode', 'oracle']), \
                mock.patch.object(r, 'load_planner') as load:
            with self.assertRaises(SystemExit):
                r.main()
            load.assert_not_called()


class RunSplitTest(unittest.TestCase):
    def run_split(self, scripts, transcripts, items=None):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        out = Path(d.name) / 'run'
        planner = ScriptedPlanner(scripts)
        envs = []

        def make_env(sid):
            envs.append(sid)
            return ToolEnvAdapter(Path(d.name), 'public_test', sid, toolenv_factory=FakeEnv)
        items = items or [item('a'), item('b')]
        with contextlib.redirect_stdout(io.StringIO()):
            report = r.run_split(items, transcripts, planner, make_env, out, split='public_test',
                                 backend={'loaded_parameters': 1}, full_split=True)
        return out, report, planner, envs

    def transcripts(self):
        return {s: {'id': s, 'request': f'asr request {s}', 'history': []} for s in 'ab'}

    def test_submission_trace_and_input_audit(self):
        out, report, planner, envs = self.run_split(
            ['{"type":"call","name":"demo.write","arguments":{"value":1}}', '{"type":"finish"}', '{"type":"finish"}'],
            self.transcripts())
        records = {json.loads(x)['id']: json.loads(x) for x in (out / 'inference/submission.jsonl').read_text().splitlines()}
        self.assertEqual(set(records), {'a', 'b'})
        self.assertEqual(records['a']['tool_calls'], [{'name': 'demo.write', 'arguments': {'value': 1}}])
        self.assertEqual(records['b']['tool_calls'], [])
        self.assertEqual(records['a']['state_hash'], HASH)
        self.assertTrue(report['trace_audit']['valid'])
        self.assertEqual((report['samples'], report['stop_reasons'], report['controller_errors']), (2, {'finished': 2}, 0))
        self.assertEqual(envs, ['a', 'b'])
        self.assertIn('asr request a', planner.seen[0])
        used = [json.loads(x) for x in (out / 'inputs_used.jsonl').read_text().splitlines()]
        self.assertEqual([u['request_sha256'] for u in used], [r.text_sha('asr request a'), r.text_sha('asr request b')])

    def test_transcripts_must_cover_exactly_the_samples(self):
        for transcripts in ({'a': {'id': 'a', 'request': 'x', 'history': []}},
                            {**self.transcripts(), 'c': {'id': 'c', 'request': 'x', 'history': []}}):
            with self.subTest(keys=sorted(transcripts)), self.assertRaises(ValueError):
                self.run_split([], transcripts)

    def test_backend_failure_is_recorded_not_hidden(self):
        out, report, planner, envs = self.run_split(
            [RuntimeError('boom'), '{"type":"finish"}'], self.transcripts())
        self.assertEqual(report['controller_errors'], 1)
        self.assertEqual(report['stop_reasons'].get('error'), 1)
        self.assertEqual(len((out / 'inference/submission.jsonl').read_text().splitlines()), 2)
        self.assertTrue(report['trace_audit']['valid'])

    def test_existing_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'inference').mkdir()
            with self.assertRaises(FileExistsError):
                r.run_split([item('a')], {'a': {'id': 'a', 'request': 'x', 'history': []}}, ScriptedPlanner([]),
                            lambda sid: None, Path(d), split='public_test', backend={}, full_split=True)


if __name__ == '__main__':
    unittest.main()
