"""Run verification: pure helpers and failure detection on a synthetic run directory."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evaluation import verify_run as v


class HelperTest(unittest.TestCase):
    def test_index_rejects_missing_duplicate_and_unexpected(self):
        rows = [{'id': 'a'}, {'id': 'b'}]
        self.assertEqual(set(v.index(rows, ['a', 'b'])), {'a', 'b'})
        for bad_rows, ids in ((rows, ['a']), (rows + [{'id': 'a'}], ['a', 'b']), (rows, ['a', 'c'])):
            with self.assertRaises(ValueError):
                v.index(bad_rows, ids)

    def test_ask_signature_only_counts_ask_calls(self):
        calls = [{'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'x.y', 'params': ['b', 'a']}},
                 {'name': 'x.y', 'arguments': {}}]
        self.assertEqual(v.ask_signature(calls), {('x.y', ('a', 'b'))})
        self.assertEqual(v.ask_signature([]), set())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def make_run(root, state_hash='h1', replay_hash='h1', stored_score=1.0):
    rec = {'id': 's1', 'tool_calls': [], 'state_hash': state_hash}
    write(root / 'inference/submission.jsonl', json.dumps(rec) + '\n')
    write(root / 'inference/trace.jsonl', '')
    write(root / 'inference/report.json', {'rows': [{'id': 's1', 'elapsed_seconds': 1.0}], 'stop_reasons': {'finished': 1}})
    write(root / 'scores/scores.json', {'rows': [{'id': 's1', 'local_score': stored_score, 'hash_match': True,
                                                  'pairs': [], 'matched_calls': 0}]})
    write(root / 'scores/replay-predictions.json', {'rows': [{'id': 's1', 'replayed_hash': replay_hash,
                                                              'schema_rejected_indices': []}],
                                                    'schema_rejections': 0, 'business_failures': 0})
    write(root / 'scores/replay-gold.json', {'rows': [{'id': 's1', 'replayed_hash': 'h1'}]})
    return {'s1': SimpleNamespace(reference=SimpleNamespace(tool_calls=[]), agent_input=SimpleNamespace(tools=[]))}


FRESH = {'local_score': 1.0, 'hash_match': True, 'pairs': [], 'matched_calls': 0}


class VerifyRunTest(unittest.TestCase):
    def run_verify(self, **kw):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(v, 'audit_trace', return_value={'valid': kw.pop('trace_ok', True)}), \
                mock.patch.object(v, 'score_sample', return_value=kw.pop('fresh', FRESH)):
            examples = make_run(Path(d), **kw)
            return v.verify_run(d, ['s1'], examples)

    def test_consistent_run_passes(self):
        summary, pred, stored = self.run_verify()
        self.assertEqual((summary['samples'], summary['hash_matches'], summary['mean_local_score']), (1, 1, 1.0))

    def test_failed_trace_audit_is_fatal(self):
        with self.assertRaisesRegex(ValueError, 'trace audit'):
            self.run_verify(trace_ok=False)

    def test_replay_hash_must_equal_submission_hash(self):
        with self.assertRaisesRegex(ValueError, 'replay hash'):
            self.run_verify(replay_hash='other')

    def test_recomputed_score_must_equal_stored_score(self):
        with self.assertRaisesRegex(ValueError, 'score mismatch'):
            self.run_verify(fresh={**FRESH, 'local_score': 0.5})


if __name__ == '__main__':
    unittest.main()
