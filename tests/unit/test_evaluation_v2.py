from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from evaluation.evaluate_run import evaluate
from evaluation.replay_predictions import read_predictions, replay_records
from lifemate.agent.controller import run_controller
from lifemate.agent.planner import LocalPlanner
from lifemate.observability.trace import TraceWriter, audit_trace, fingerprint
from lifemate.observability.provenance import collect_provenance, source_provenance


CALL = {'name':'demo.write','arguments':{'value':1}}
TOOLS = [{'name':'demo.write','parameters':{'type':'object','properties':{'value':{'type':'integer'}}}}]


class ReplayTests(unittest.TestCase):
    def test_rejected_calls_preserved_one_session_order(self):
        calls = [deepcopy(CALL), {'name':'bad.tool','arguments':{}}]
        records = [{'id':'a','tool_calls':calls,'state_hash':'a'*64}]
        opened = []
        class Env:
            def __init__(self, root, split, sid): opened.append(sid); self.calls=[]
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def call(self, name, args):
                self.calls.append({'name':name,'arguments':args})
                return {'ok':False} if name=='bad.tool' else {'ok':True,'done':False}
            def finalize(self): return {'tool_calls':self.calls,'state_hash':'a'*64}
        ex = NS(reference=NS(tool_calls=[CALL]))
        with patch('evaluation.replay_predictions.selected_examples',return_value={'a':ex}):
            rows = replay_records('unused', records, env_factory=Env)
        self.assertEqual(opened, ['a'])
        self.assertTrue(rows[0]['ledger_matches'])
        self.assertEqual(rows[0]['schema_rejected_indices'],[1])
        self.assertEqual(rows[0]['business_failure_indices'],[0])
        self.assertEqual(records[0]['tool_calls'], calls)

    def test_evaluation_rejects_unreproducible_missing_duplicate_and_gold_rejected(self):
        records=[{'id':'a','tool_calls':[CALL],'state_hash':'a'*64}]
        examples={'a':NS(agent_input=NS(tools=TOOLS),reference=NS(tool_calls=[CALL]))}
        row={'id':'a','hash_reproduced':True,'ledger_matches':True,'replayed_hash':'a'*64,
             'schema_rejected_indices':[]}
        pred={'mode':'predictions','rows':[row]}
        gold={'mode':'gold','rows':[deepcopy(row)]}
        self.assertEqual(evaluate(records,examples,pred,gold)['mean_local_score'],1)
        for mutate in ['hash','ledger','missing','duplicate','gold_rejected']:
            pp,gg=deepcopy(pred),deepcopy(gold)
            if mutate=='hash': pp['rows'][0]['hash_reproduced']=False
            if mutate=='ledger': pp['rows'][0]['ledger_matches']=False
            if mutate=='missing': pp['rows']=[]
            if mutate=='duplicate': pp['rows']*=2
            if mutate=='gold_rejected': gg['rows'][0]['schema_rejected_indices']=[0]
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                evaluate(records,examples,pp,gg)

    def test_submission_strict_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'submission.jsonl'
            record={'id':'a','tool_calls':[CALL],'state_hash':'a'*64}
            for text in [json.dumps(record)+'\n'+json.dumps(record),
                         json.dumps(record).replace('"value": 1','"value": NaN'),
                         '{"id":"a","id":"b"}', '', json.dumps(record)+'\n\n']:
                path.write_text(text,encoding='utf-8')
                with self.subTest(text=text[:20]), self.assertRaises(ValueError): read_predictions(path)


class ControllerTraceTests(unittest.TestCase):
    def test_trace_preserves_decisions_messages_and_business_failure(self):
        outputs=['bad json',json.dumps({'type':'call','name':'demo.write','arguments':{}}),
                 json.dumps({'type':'call',**CALL}),'{"type":"finish"}']
        class Planner:
            def __init__(self): self.outputs=iter(outputs); self.inputs=[]
            def generate(self,messages,seconds): self.inputs.append(deepcopy(messages)); return next(self.outputs)
        class Env:
            def validate(self,name,args): return {'ok':'value' in args,'error':'required value'}
            def call(self,name,args): return {'ok':True,'done':False,'message':'synthetic failure'}
        payload={'tools':TOOLS}
        before, after = Planner(), Planner()
        original = run_controller(payload,Env(),before)
        events=[]
        observed = run_controller(payload,Env(),after,trace=lambda event,**data:events.append((event,data)))
        self.assertEqual(before.inputs,after.inputs)
        self.assertEqual((original.stop,original.calls,original.errors,original.decisions),
                         (observed.stop,observed.calls,observed.errors,observed.decisions))
        self.assertEqual(sum(e=='parse_error' for e,_ in events),1)
        result=next(data for event,data in events if event=='call_result')
        self.assertTrue(result['business_failure'])
        self.assertFalse(result['schema_rejected'])

    def test_deadline_backend_failure_and_partial_counts(self):
        class Planner:
            last_generation={'finish_reason':'token_limit','new_tokens':384}
            def generate(self,*args): return 'truncated'
        events=[]
        result=run_controller({'tools':[]},NS(),Planner(),max_repairs=0,
                              trace=lambda event,**data:events.append((event,data)))
        self.assertEqual(result.stop,'invalid_action')
        self.assertEqual(events[1][1]['generation']['finish_reason'],'token_limit')
        class Timeout:
            def generate(self,*args): raise TimeoutError('synthetic')
        events=[]
        self.assertEqual(run_controller({'tools':[]},NS(),Timeout(),
                         trace=lambda e,**d:events.append((e,d))).stop,'deadline')
        self.assertEqual(events[1][1]['error_category'],'deadline')
        class Error:
            def generate(self,*args): raise RuntimeError('synthetic')
        with self.assertRaises(RuntimeError) as caught:
            run_controller({'tools':[]},NS(),Error())
        self.assertEqual(caught.exception.controller_outcome.decisions,1)

    def test_call_failure_is_indeterminate_not_retried(self):
        class Planner:
            def generate(self,*args): return json.dumps({'type':'call',**CALL})
        class Env:
            count=0
            def validate(self,*args): return {'ok':True}
            def call(self,*args): self.count+=1; raise RuntimeError('unknown side effect')
        env=Env(); events=[]
        with self.assertRaises(RuntimeError):
            run_controller({'tools':TOOLS},env,Planner(),trace=lambda e,**d:events.append((e,d)))
        self.assertEqual(env.count,1)
        self.assertTrue(events[-1][1]['indeterminate'])


class GenerationDiagnosticsTests(unittest.TestCase):
    def planner(self, *, new_tokens=2, context_tokens=10, tokens=(1,0)):
        class Tensor:
            shape=(1,1)
            def __getitem__(self,key): return self
        class Batch(dict):
            def to(self,device): return self
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs): return 'prompt'
            def __call__(self,*args,**kwargs): return Batch(input_ids=Tensor(),attention_mask=Tensor())
            def decode(self,ids,**kwargs): return ''.join('action' for _ in ids)
        class Model:
            generation_config=NS(eos_token_id=0)
            def __init__(self): self.tokens=iter(tokens)
            def __call__(self,**kwargs):
                token=next(self.tokens)
                class Logits(Tensor):
                    def argmax(self,**kwargs): return NS(item=lambda:token)
                return NS(logits=Logits(),past_key_values='cache')
        planner=LocalPlanner.__new__(LocalPlanner)
        planner.model=Model(); planner.tokenizer=Tokenizer()
        planner.new_tokens=new_tokens; planner.context_tokens=context_tokens
        torch=NS(inference_mode=nullcontext,tensor=lambda *a,**kw:Tensor(),
                 ones=lambda *a,**kw:Tensor(),long='long')
        return planner,torch

    def test_real_loop_eos_token_limit_context_limit_and_deadline_without_gpu(self):
        for reason, kwargs in [('eos',{}),('token_limit',{'new_tokens':1}),
                               ('context_limit',{'context_tokens':2})]:
            planner,torch=self.planner(**kwargs)
            with self.subTest(reason=reason), patch.dict('sys.modules',torch=torch):
                if reason=='context_limit':
                    with self.assertRaises(ValueError): planner.generate([],10)
                else: self.assertEqual(planner.generate([],10),'action')
                self.assertEqual(planner.last_generation['finish_reason'],reason)
        planner,torch=self.planner()
        with patch.dict('sys.modules',torch=torch), patch('lifemate.agent.planner.time.monotonic',side_effect=[0,0,2]):
            with self.assertRaises(TimeoutError): planner.generate([],1)
        self.assertEqual(planner.last_generation['finish_reason'],'deadline')


class ProvenanceTests(unittest.TestCase):
    def test_model_revision_from_download_metadata_and_unknown_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text('{}',encoding='utf-8')
            result=collect_provenance(root,config={},model_path=root)
            self.assertIsNone(result['model']['revision'])
            path=root/'.cache/huggingface/download/config.json.metadata'
            path.parent.mkdir(parents=True); path.write_text('a'*40+'\netag\n1',encoding='utf-8')
            self.assertEqual(collect_provenance(root,config={},model_path=root)['model']['revision'],'a'*40)
            child=root/'unpacked-wheel';child.mkdir()
            with patch('lifemate.observability.provenance.subprocess.check_output',return_value=str(root).encode()):
                self.assertTrue(source_provenance(child)['unavailable'])
                self.assertIsNone(source_provenance(child)['commit'])


if __name__=='__main__': unittest.main()
