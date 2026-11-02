from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.analysis import main


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'data'
        (self.data / 'train').mkdir(parents=True)
        self.run = self.root / 'run'
        self.run.mkdir()
        call = {'name': 'notes.create', 'arguments': {'text': 'synthetic'}}
        self.gold = [{'id': 'a', 'history': [], 'label': {'tool_calls': [call, call], 'final_state': {'notes': {}}}},
                     {'id': 'b', 'history': [], 'label': {'tool_calls': [], 'final_state': {}}}]
        self.pred = [{'id': 'a', 'tool_calls': [deepcopy(call)], 'state_hash': 'a'*64},
                     {'id': 'b', 'tool_calls': [], 'state_hash': 'b'*64}]
        self.metrics = [{'id': x, 'full_state_match': x == 'b', 'stop': 'finished',
                         'errors': 0, 'elapsed_seconds': 1.0} for x in ['a', 'b']]
        self.smoke = self.root / 'smoke.json'
        self.write()

    def write(self):
        (self.data/'train/train.jsonl').write_text('\n'.join(map(json.dumps, self.gold)), encoding='utf-8')
        (self.run/'submission.jsonl').write_text('\n'.join(map(json.dumps, self.pred)), encoding='utf-8')
        (self.run/'report.json').write_text(json.dumps({'samples': len(self.metrics), 'rows': self.metrics}), encoding='utf-8')
        self.smoke.write_text(json.dumps({'rows': self.metrics}), encoding='utf-8')

    def execute(self, *extra):
        with redirect_stdout(io.StringIO()):
            main(['--data-root', str(self.data), '--run-dir', str(self.run),
                  '--smoke-report', str(self.smoke), *extra])
        return json.loads((self.run/'analysis.json').read_text(encoding='utf-8'))

    def test_matching_does_not_reuse_one_call(self):
        result = self.execute()
        self.assertEqual(result['totals']['exact_argument_matches'], 1)
        self.assertEqual(result['totals']['exact_unordered_call_samples'], 1)
        self.assertEqual(result['totals']['samples_with_missing_tool_names'], 1)
        self.assertEqual(result['totals']['empty_gold_correct'], 1)

    def test_argument_and_name_matching_are_separate(self):
        self.gold[0]['label']['tool_calls'] = self.gold[0]['label']['tool_calls'][:1]
        self.pred[0]['tool_calls'][0]['arguments'] = {'text': 'different'}
        self.write()
        result = self.execute()
        self.assertEqual(result['totals']['name_matches'], 1)
        self.assertEqual(result['totals']['exact_argument_matches'], 0)

    def test_duplicate_or_missing_prediction_ids_fail(self):
        for pred in [self.pred + [deepcopy(self.pred[0])], self.pred[:1]]:
            with self.subTest(predictions=len(pred)):
                (self.run/'submission.jsonl').write_text('\n'.join(map(json.dumps, pred)), encoding='utf-8')
                with self.assertRaises(ValueError): self.execute()
                self.assertFalse((self.run/'analysis.json').exists())

    def test_report_duplicates_and_missing_references(self):
        self.metrics.append(deepcopy(self.metrics[0])); self.write()
        with self.assertRaises(ValueError): self.execute()
        self.metrics.pop(); self.gold.pop(); self.write()
        with self.assertRaises(ValueError): self.execute()

    def test_output_cannot_overwrite_or_escape(self):
        self.execute()
        original = (self.run/'analysis.json').read_bytes()
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual((self.run/'analysis.json').read_bytes(), original)
        with self.assertRaises(ValueError): self.execute('--output-name', '../escape.json')

    def test_hash_replay_one_session_per_id_preserves_predictions(self):
        opened = []
        class Fake:
            def __init__(self, root, split, id): opened.append((split, id)); self.id = id
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def call(self, name, arguments): pass
            def finalize(self): return {'state_hash': self.id*64}
        before = (self.run/'submission.jsonl').read_bytes()
        with patch('lifemate.tools.env_adapter.ToolEnvAdapter', Fake):
            result = self.execute('--verify-gold-hashes')
        self.assertEqual(opened, [('train','a'), ('train','b')])
        self.assertEqual(result['totals']['gold_hash_matches'], 2)
        self.assertEqual((self.run/'submission.jsonl').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
