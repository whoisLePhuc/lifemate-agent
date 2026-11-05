import json
import unittest

from lifemate.agent.controller import run_controller


def call(name, **arguments):
    return json.dumps({'type': 'call', 'name': name, 'arguments': arguments})


class Planner:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def generate(self, messages, seconds):
        self.messages.append(messages)
        return next(self.actions)


class Env:
    def __init__(self, responses=()):
        self.calls = []
        self.responses = iter(responses)

    def validate(self, name, arguments):
        return {'ok': True}

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return next(self.responses, {'ok': True, 'done': True})


class RepeatedReadTests(unittest.TestCase):
    def setUp(self):
        self.payload = {'request': 'test', 'history': [], 'context': {}, 'tools': [
            {'name': 'mail.search', 'kind': 'read'},
            {'name': 'mail.delete', 'kind': 'write'},
        ]}
        self.read = call('mail.search', a=1, b=2)
        self.reordered = call('mail.search', b=2, a=1)
        self.finish = json.dumps({'type': 'finish'})

    def run_actions(self, actions, responses=(), *, policy='repeat_reads_v1', **options):
        planner, env, trace = Planner(actions), Env(responses), []
        outcome = run_controller(self.payload, env, planner, policy=policy,
                                 trace=lambda event, **data: trace.append(event), **options)
        return outcome, env, planner, trace

    def test_reordered_arguments_are_blocked_and_feedback_reaches_planner(self):
        outcome, env, planner, trace = self.run_actions([self.read, self.reordered, self.finish])
        self.assertEqual(outcome.stop, 'finished')
        self.assertEqual(len(env.calls), 1)
        self.assertIn('repeated_read_blocked', trace)
        self.assertIn('Truy vấn này đã thành công', planner.messages[2][1]['content'])
        self.assertIn('observation', planner.messages[2][1]['content'])

    def test_write_invalidates_cached_reads(self):
        outcome, env, _, _ = self.run_actions(
            [self.read, call('mail.delete', id='a'), self.read, self.finish])
        self.assertEqual(outcome.stop, 'finished')
        self.assertEqual(len(env.calls), 3)

    def test_failed_read_can_be_retried_and_different_arguments_are_distinct(self):
        outcome, env, _, _ = self.run_actions(
            [self.read, self.read, call('mail.search', a=2), self.finish],
            [{'ok': False}, {'ok': True}, {'ok': True}])
        self.assertEqual(outcome.stop, 'finished')
        self.assertEqual(len(env.calls), 3)

    def test_repeat_limit_stops_without_dispatch(self):
        outcome, env, _, _ = self.run_actions([self.read, self.read, self.read], max_repairs=1)
        self.assertEqual(outcome.stop, 'repeat_read_limit')
        self.assertEqual(len(env.calls), 1)

    def test_default_policy_preserves_repeated_reads(self):
        planner, env = Planner([self.read, self.read, self.finish]), Env()
        run_controller(self.payload, env, planner)
        self.assertEqual(len(env.calls), 2)

    def test_observation_recovery_allows_one_retry_without_dispatching_duplicate(self):
        outcome, env, planner, _ = self.run_actions(
            [self.read, self.read, self.read], [{'ok': True, 'messages': []}],
            policy='repeat_reads_observation')
        self.assertEqual(outcome.stop, 'repeat_read_limit')
        self.assertEqual(len(env.calls), 1)
        feedback = json.loads(planner.messages[2][1]['content'])['current_turn_events'][-1]
        self.assertEqual(feedback['previous_observation']['messages'], [])
        self.assertFalse(feedback['executed'])

    def test_options_include_only_schema_linked_clarification(self):
        self.payload['tools'][1]['parameters'] = {
            'required': ['message_id', 'other_id'],
            'properties': {'message_id': {'x-ref': 'mail.messages'},
                           'other_id': {'x-ref': 'other.messages'}},
        }
        self.payload['tools'].append({'name': 'sys.ask_missing_info', 'kind': 'meta'})
        outcome, env, planner, _ = self.run_actions(
            [self.read, self.read, self.finish], [{'ok': True, 'messages': []}],
            policy='repeat_reads_options')
        self.assertEqual(outcome.stop, 'finished')
        self.assertEqual(len(env.calls), 1)
        feedback = json.loads(planner.messages[2][1]['content'])['current_turn_events'][-1]
        asks = [option['arguments'] for option in feedback['next_options'] if isinstance(option, dict)]
        self.assertEqual(asks, [{'tool_name': 'mail.delete', 'params': ['message_id']}])

    def test_xref_reports_observed_candidates_without_selecting_one(self):
        self.payload['tools'][1]['parameters'] = {'properties': {
            'message_id': {'x-ref': 'mail.messages'},
            'contact_id': {'x-ref': 'contacts.messages'},
        }}
        for count in (0, 1, 2):
            with self.subTest(count=count):
                observation = {'ok': True, 'messages': [{'id': str(i)} for i in range(count)]}
                _, env, planner, _ = self.run_actions(
                    [self.read, self.finish], [observation], policy='xref_observation_v1')
                event = json.loads(planner.messages[1][1]['content'])['current_turn_events'][-1]
                self.assertEqual(event['reference_sources'], [
                    {'tool': 'mail.delete', 'field': 'message_id',
                     'candidate_id_path': 'messages[].id', 'candidate_count': count}])
                self.assertEqual(len(env.calls), 1)


if __name__ == '__main__':
    unittest.main()
