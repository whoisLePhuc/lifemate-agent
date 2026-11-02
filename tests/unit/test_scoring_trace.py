import json
from pathlib import Path
import tempfile
import unittest

from evaluation.scoring import argument_match, match_calls, score_sample, token_f1
from lifemate.observability.trace import TraceWriter, audit_trace, fingerprint

TOOLS = [{'name': 'demo.write', 'parameters': {'type': 'object', 'properties': {
    'body': {'type': 'string', 'x-match': 'fuzzy'}, 'id': {'type': 'string'}}}}]
def call(body, **extra):
    return {'name': 'demo.write', 'arguments': {'body': body, **extra}}


class ScoringTests(unittest.TestCase):
    def test_fuzzy_normalization_and_boundary(self):
        self.assertEqual(token_f1('CAFE\u0301', 'café'), 1)
        self.assertEqual(token_f1('café', 'cafe'), 0)
        self.assertEqual(token_f1('a a b', 'a b b'), 2/3)
        self.assertEqual(token_f1('', ''), 1)
        self.assertEqual(token_f1('', 'a'), 0)
        # 3 common tokens, 5 tokens each => exactly .6 does not pass.
        self.assertEqual(match_calls([call('a b c d e')], [call('a b c x y')], TOOLS)['matched_calls'], 0)

    def test_maximum_matching_not_greedy(self):
        # p0 fits g0/g1; p1 fits only g0. Requires reassigning p0.
        result = match_calls([call('a b c'), call('a b d')],
                             [call('a b c'), call('a c e')], TOOLS)
        self.assertEqual(result['matched_calls'], 2)
        self.assertEqual(match_calls([call('a b c')], [call('a b c')]*2, TOOLS)['matched_calls'], 1)

    def test_exact_fields_keys_types_and_arrays(self):
        self.assertEqual(match_calls([call('same', id='x')], [call('same', id='y')], TOOLS)['matched_calls'], 0)
        self.assertEqual(match_calls([call('same', extra=1)], [call('same')], TOOLS)['matched_calls'], 0)
        self.assertFalse(argument_match({'x': True}, {'x': 1}, {}))
        self.assertFalse(argument_match([1, 2], [2, 1], {}))
        self.assertTrue(argument_match({'x': 1}, {'x': 1.0}, {}))

    def test_gate_empty_gold_rejected_and_unknown(self):
        self.assertEqual(score_sample([call('a')], [call('a')], TOOLS, predicted_hash='x', gold_hash='y')['local_score'], 0)
        self.assertEqual(score_sample([], [], TOOLS, predicted_hash='x', gold_hash='y')['local_score'], 1)
        self.assertEqual(score_sample([call('a')], [], TOOLS, predicted_hash='x', gold_hash='x')['local_score'], 0)
        self.assertEqual(match_calls([call('a')], [call('a')], TOOLS, rejected_indices=[0])['matched_calls'], 0)
        self.assertEqual(match_calls([call('a')], [call('a')], [])['matched_calls'], 0)


class TraceTests(unittest.TestCase):
    def test_complete_trace_and_interruptions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            action = {'type': 'call', **call('synthetic')}
            record = {'id': 'a', 'tool_calls': [call('synthetic')], 'state_hash': 'a'*64}
            with TraceWriter(path, 'run') as writer:
                writer.emit('run_start')
                emit = writer.bind('a')
                emit('sample_start')
                emit('call_start', action=action)
                emit('call_result', action=action, observation={'ok': True, 'done': False})
                emit('sample_end', record_sha256=fingerprint(record))
                writer.emit('run_end')
            self.assertTrue(audit_trace(path, [record])['valid'])
            tampered = {**record, 'tool_calls': []}
            self.assertIn('a:ledger_mismatch', audit_trace(path, [tampered])['issues'])
            self.assertIn('a:record_fingerprint_mismatch', audit_trace(path, [tampered])['issues'])
            lines = path.read_text(encoding='utf-8').splitlines()
            path.write_text('\n'.join(lines[:3])+'\n', encoding='utf-8')
            report = audit_trace(path, [record])
            self.assertFalse(report['valid'])
            self.assertIn('a:indeterminate_call', report['issues'])
            self.assertIn('missing_run_end', report['issues'])
            with self.assertRaises(FileExistsError): TraceWriter(path, 'new')


if __name__ == '__main__': unittest.main()
