import unittest
from types import SimpleNamespace

from lifemate.agent.controller import run_controller
from lifemate.agent.prompt import SYSTEM, SYSTEM_EN
from lifemate.agent.protocol import (NATIVE_SYSTEM, NATIVE_SYSTEM_EN, build_messages, parse_native_xml,
                                     parse_output, system_prompt)

TOOLS = {
    'sms.send': {'name': 'sms.send', 'kind': 'write', 'parameters': {'type': 'object', 'properties': {
        'contact_id': {'type': 'string'}, 'content': {'type': 'string'}, 'count': {'type': 'integer'},
        'urgent': {'type': 'boolean'}, 'tags': {'type': 'array'}, 'meta': {'type': 'object'},
        'maybe': {'type': ['integer', 'null']}}}},
    'alarm.list': {'name': 'alarm.list', 'kind': 'read', 'parameters': {'type': 'object', 'properties': {}}},
}


def call(*params, name='sms.send'):
    body = ''.join(f'<parameter={key}>\n{value}\n</parameter>\n' for key, value in params)
    return f'<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>'


class XmlParserTests(unittest.TestCase):
    def parse(self, text):
        return parse_native_xml(text, TOOLS)

    def test_parses_strings_and_typed_values(self):
        action = self.parse(call(('contact_id', 'c_1'), ('content', 'Tôi đến muộn\ndòng hai'), ('count', '3'),
                                 ('urgent', 'True'), ('tags', '["a", "b"]'), ('meta', '{"k": 1}'), ('maybe', 'null')))
        self.assertEqual(action, {'type': 'call', 'name': 'sms.send', 'arguments': {
            'contact_id': 'c_1', 'content': 'Tôi đến muộn\ndòng hai', 'count': 3, 'urgent': True,
            'tags': ['a', 'b'], 'meta': {'k': 1}, 'maybe': None}})

    def test_string_parameter_stays_a_string_even_if_it_looks_like_json(self):
        self.assertEqual(self.parse(call(('content', '123')))['arguments'], {'content': '123'})
        self.assertEqual(self.parse(call(('content', 'true')))['arguments'], {'content': 'true'})

    def test_no_arguments_and_reasoning_prefix(self):
        self.assertEqual(self.parse(call(name='alarm.list'))['arguments'], {})
        self.assertEqual(self.parse('Tôi sẽ tra cứu.\n\n' + call(name='alarm.list'))['name'], 'alarm.list')

    def test_plain_text_is_finish(self):
        self.assertEqual(self.parse('Đã xong.'), {'type': 'finish'})

    def test_rejections(self):
        bad = [
            '', '   ', '{"type":"finish"}', '<function=x></function>',
            call(('count', 'abc')),                                  # not JSON for an integer
            call(('urgent', 'maybe')),
            call(('content', 'a'), ('content', 'b')),                # duplicate parameter
            call(name='alarm.list') + call(name='alarm.list'),       # two calls
            call(name='alarm.list') + '\nxong rồi',                  # suffix after the call
            '<tool_call>\n<function=alarm.list>\n</function>',       # no closing tag
            '<tool_call>\n<function=alarm.list>\njunk\n</function>\n</tool_call>',
            '<tool_call>\nnot a function\n</tool_call>',
            '<think>x</think>' + call(name='alarm.list'),            # think leaked into action text
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_unknown_tool_keeps_raw_strings_for_validator_to_reject(self):
        self.assertEqual(parse_native_xml(call(('x', '1'), name='nope.tool'), TOOLS)['arguments'], {'x': '1'})

    def test_parse_output_dispatch(self):
        text = call(name='alarm.list')
        self.assertEqual(parse_output(text, 'qwen35_native', TOOLS)['name'], 'alarm.list')
        with self.assertRaises(ValueError):
            parse_output(text, 'qwen_native')   # legacy JSON parser must not accept XML


class PromptTests(unittest.TestCase):
    def test_language_selection_and_defaults(self):
        self.assertEqual(system_prompt('custom_json'), SYSTEM)
        self.assertEqual(system_prompt('custom_json', lang='en'), SYSTEM_EN)
        self.assertEqual(system_prompt('qwen35_native'), NATIVE_SYSTEM)
        self.assertEqual(system_prompt('qwen35_native', lang='en'), NATIVE_SYSTEM_EN)
        with self.assertRaises(ValueError): system_prompt('custom_json', lang='fr')
        with self.assertRaises(ValueError): system_prompt('custom_json', 'grounded_v1', 'en')
        with self.assertRaises(ValueError): system_prompt('qwen35_native', 'grounded_v1')

    def test_english_prompts_keep_the_same_instruction_set(self):
        for vi, en in ((SYSTEM, SYSTEM_EN), (NATIVE_SYSTEM, NATIVE_SYSTEM_EN)):
            self.assertEqual(len(vi.splitlines()), len(en.splitlines()))
        for tool in ('sys.ask_missing_info', 'context.current_time'):
            self.assertIn(tool, SYSTEM_EN); self.assertIn(tool, NATIVE_SYSTEM_EN)
        self.assertIn('"type":"finish"', SYSTEM_EN)

    def payload(self):
        return {'request': 'r', 'context': {}, 'history': [], 'tools': list(TOOLS.values())}

    def test_messages_use_selected_system_prompt(self):
        self.assertEqual(build_messages(self.payload(), [], 'custom_json', 'off', 'en')[0]['content'], SYSTEM_EN)
        self.assertEqual(build_messages(self.payload(), [], 'custom_json')[0]['content'], SYSTEM)

    def test_qwen35_history_passes_arguments_as_mapping(self):
        event = {'action': {'type': 'call', 'name': 'sms.send', 'arguments': {'count': 2}}, 'observation': {'ok': True}}
        rendered = build_messages(self.payload(), [event], 'qwen35_native')
        self.assertEqual(rendered[2]['tool_calls'][0]['function']['arguments'], {'count': 2})
        legacy = build_messages(self.payload(), [event], 'qwen_native')
        self.assertIsInstance(legacy[2]['tool_calls'][0]['function']['arguments'], str)


class Planner:
    def __init__(self, outputs):
        self.outputs, self.calls, self.last_generation = list(outputs), [], {}

    def generate(self, messages, seconds, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class Env:
    def __init__(self): self.calls = []
    def validate(self, name, arguments): return {'ok': True}
    def call(self, name, arguments):
        self.calls.append((name, arguments)); return {'ok': True, 'done': True}


class ControllerTests(unittest.TestCase):
    payload = {'request': 'r', 'context': {}, 'history': [], 'tools': list(TOOLS.values())}

    def run_xml(self, outputs, lang='vi'):
        planner, env = Planner(outputs), Env()
        result = run_controller(self.payload, env, planner, protocol='qwen35_native', lang=lang)
        return result, env, planner

    def test_call_then_text_finish(self):
        result, env, planner = self.run_xml([call(('content', 'hi'), ('count', '2')), 'Xong.'])
        self.assertEqual(result.stop, 'finished')
        self.assertEqual(env.calls, [('sms.send', {'content': 'hi', 'count': 2})])
        self.assertTrue(all('tools' in kwargs for kwargs in planner.calls))

    def test_malformed_xml_is_repaired_not_dispatched(self):
        result, env, _ = self.run_xml(['<tool_call>\n<function=sms.send>', call(name='alarm.list'), 'ok'])
        self.assertEqual(result.stop, 'finished')
        self.assertEqual(env.calls, [('alarm.list', {})])

    def test_token_limit_output_is_never_dispatched(self):
        planner = Planner([call(name='alarm.list')]); planner.last_generation = {'finish_reason': 'token_limit'}
        env = Env()
        result = run_controller(self.payload, env, planner, protocol='qwen35_native', max_repairs=0)
        self.assertEqual(env.calls, [])
        self.assertNotEqual(result.stop, 'finished')


class ThinkingPlannerTests(unittest.TestCase):
    """Budget forcing logic without loading a model."""

    def make(self, budget):
        from lifemate.agent.qwen35_planner import Qwen35Planner
        planner = object.__new__(Qwen35Planner)
        planner.thinking, planner.think_budget = True, budget
        planner.think_end_id, planner._close_ids, planner._force = 9, [1, 9, 2], []
        planner.last_generation = {}
        planner._generated, planner._active_tools = [], None
        planner.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: ' '.join(map(str, ids)))
        return planner

    def test_forces_close_once_budget_is_spent(self):
        planner, generated, out = self.make(3), [], []
        for _ in range(7):
            token = planner._choose(5, generated); generated.append(token); out.append(token)
        self.assertEqual(out, [5, 5, 5, 1, 9, 2, 5])
        self.assertTrue(planner.last_generation['thinking_forced'])

    def test_model_closing_itself_is_not_forced(self):
        planner, generated = self.make(3), [5]
        generated.append(planner._choose(9, generated))
        generated.append(planner._choose(5, generated))
        self.assertEqual(generated, [5, 9, 5]); self.assertNotIn('thinking_forced', planner.last_generation)

    def test_answer_is_text_after_think_end_only(self):
        planner = self.make(3)
        planner._generated = [5, 5, 9, 7, 8]
        self.assertEqual(planner._decode_generated(), '7 8')
        planner._generated = [5, 5]
        self.assertEqual(planner._decode_generated(), '')   # still thinking: nothing to dispatch


if __name__ == '__main__':
    unittest.main()
