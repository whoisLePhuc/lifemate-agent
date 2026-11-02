from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from lifemate_agent.text import parse_action,run_controller,text_input,messages


class Planner:
    def __init__(self, outputs):self.outputs=iter(outputs);self.inputs=[]
    def generate(self,msg,seconds):self.inputs.append(msg);return next(self.outputs)


class Env:
    def __init__(self):self.calls=[]
    def validate(self,name,args):return {'ok':'value' in args,'error':'required value'}
    def call(self,name,args):self.calls.append(name);return {'ok':True,'done':True}


class TextTests(unittest.TestCase):
    def payload(self):return {'tools':[{'name':'demo.write'},{'name':'sys.ask_missing_info'}]}
    def test_input_has_transcripts_but_no_gold(self):
        item=NS(context={},history=[{'role':'user','audio':Path('x')},{'role':'assistant','text':'old','tool_results':[{'ok':True}]}],tools=[])
        payload=text_input(item,'current',('prior',))
        self.assertEqual(payload['history'][0],{'role':'user','text':'prior'})
        self.assertEqual(set(payload),{'request','context','history','tools'})
        self.assertNotIn('audio_text',str(messages(payload,[])))
        with self.assertRaises(ValueError):text_input(item,'current',())
    def test_strict_action_parser(self):
        for text in ['```json\n{}\n```','{}','{"type":"finish","extra":1}',
                     '{"type":"finish","type":"finish"}','{"type":"call","name":"x","arguments":{"a":NaN}}']:
            with self.assertRaises(ValueError):parse_action(text)
    def test_closed_loop_observation(self):
        env=Env();p=Planner(['{"type":"call","name":"demo.write","arguments":{"value":1}}','{"type":"finish"}'])
        r=run_controller(self.payload(),env,p)
        self.assertEqual(r.stop,'finished');self.assertEqual(env.calls,['demo.write'])
        self.assertIn('observation',str(p.inputs[1]))
    def test_invalid_then_repair_no_dispatch(self):
        env=Env();p=Planner(['bad','{"type":"call","name":"demo.write","arguments":{}}','{"type":"finish"}'])
        r=run_controller(self.payload(),env,p)
        self.assertEqual(r.errors,2);self.assertEqual(env.calls,[])
    def test_allowlist_and_repair_limit(self):
        env=Env();p=Planner(['{"type":"call","name":"outside","arguments":{}}']*3)
        self.assertEqual(run_controller(self.payload(),env,p).stop,'validation_limit')
        self.assertEqual(env.calls,[])
    def test_ask_is_terminal(self):
        env=Env();p=Planner(['{"type":"call","name":"sys.ask_missing_info","arguments":{"value":1}}'])
        self.assertEqual(run_controller(self.payload(),env,p).stop,'asked_missing_info')
    def test_limits_and_timeout(self):
        env=Env();p=Planner(['{"type":"call","name":"demo.write","arguments":{"value":1}}']*3)
        self.assertEqual(run_controller(self.payload(),env,p,max_calls=1).stop,'max_calls')
        self.assertEqual(len(env.calls),1)
        class Timeout:
            def generate(self,*args):raise TimeoutError()
        self.assertEqual(run_controller(self.payload(),Env(),Timeout()).stop,'deadline')
        class Late:
            def generate(self,*args):return '{"type":"call","name":"demo.write","arguments":{"value":1}}'
        env=Env()
        with patch('lifemate_agent.text.time.monotonic',side_effect=[0,0,31,31]):
            r=run_controller(self.payload(),env,Late())
        self.assertEqual(r.stop,'deadline');self.assertEqual(env.calls,[])


if __name__=='__main__':unittest.main()
