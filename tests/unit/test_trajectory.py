from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import json
import unittest

from training.trajectories import inner_split,trajectory_rows
from training.loss_mask import encode_action,validate_mask,collate_actions,SequenceBudgetError


CALL={'name':'notes.create','arguments':{'title':'TARGET'}}
def example(calls=None,results=None):
    calls=[deepcopy(CALL)] if calls is None else calls
    results=[{'ok':True,'id':'PRIOR_RESULT'}] if results is None else results
    return NS(agent_input=NS(id='a',context={},tools=[{'name':'notes.create'},{'name':'sys.ask_missing_info'}],
      history=[],audio=Path('unused.wav')),reference=NS(audio_text='request',history_audio_text=(),
      tool_calls=calls,tool_results=results,final_state={'DO_NOT_USE':'FINAL_STATE_SECRET'}))


class Tokenizer:
    eos_token='~';eos_token_id=126;pad_token_id=126
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        return ''.join(m['role']+':'+m['content']+'~' for m in messages)+('assistant:' if add_generation_prompt else '')
    def __call__(self,text,**kwargs):return {'input_ids':list(text.encode('utf-8'))}
    def decode(self,ids,**kwargs):return bytes(ids).decode('utf-8')


class TrajectoryTests(unittest.TestCase):
    def test_prefix_only_and_no_state_or_future_observation(self):
        calls=[deepcopy(CALL),{'name':'notes.create','arguments':{'title':'FUTURE_TARGET'}}]
        rows=list(trajectory_rows(example(calls,[{'ok':True,'id':'PAST_RESULT'},{'ok':True,'id':'FUTURE_OBSERVATION'}])))
        first=json.loads(rows[0]['prompt'][1]['content']);second=json.loads(rows[1]['prompt'][1]['content'])
        self.assertEqual(first['current_turn_events'],[])
        self.assertEqual(len(second['current_turn_events']),1)
        self.assertNotIn('FUTURE_OBSERVATION',str(rows[1]['prompt']))
        self.assertNotIn('FUTURE_TARGET',str(rows[0]['prompt']))
        self.assertNotIn('FINAL_STATE_SECRET',str(rows))
        self.assertEqual(json.loads(rows[-1]['completion'][0]['content']),{'type':'finish'})
    def test_empty_and_ask_termination(self):
        rows=list(trajectory_rows(example([],[])));self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['target_type'],'finish')
        ask={'name':'sys.ask_missing_info','arguments':{'tool_name':'notes.create','params':['title']}}
        self.assertEqual(len(list(trajectory_rows(example([ask],[{'ok':True}])))),1)
        with self.assertRaises(ValueError):list(trajectory_rows(example([ask,CALL],[{'ok':True}]*2)))
        with self.assertRaises(ValueError):list(trajectory_rows(example([CALL],[])))
    def test_heldout_groups_and_seen_groups_stay_together(self):
        m={'train_ids':['a','b','c','d','e'],'dev_ids':['z'],'groups':[['a','b'],['c'],['d'],['e'],['z']]}
        split=inner_split(m,validation_fraction=.2,seen_ids=['a'])
        self.assertTrue({'a','b'}<=set(split['train_ids']))
        self.assertFalse(set(split['validation_ids'])&{'a','b','z'})
        reversed_m={**m,'train_ids':list(reversed(m['train_ids'])),'groups':list(reversed(m['groups']))}
        self.assertEqual(split,inner_split(reversed_m,validation_fraction=.2,seen_ids=['a']))
        bad=deepcopy(m);bad['groups']=[['a','z'],['b','c','d','e']]
        with self.assertRaises(ValueError):inner_split(bad)
        with self.assertRaises(ValueError):inner_split(m,seen_ids=['z'])


class MaskTests(unittest.TestCase):
    def row(self):return list(trajectory_rows(example()))[0]
    def test_only_completion_and_eos_loss(self):
        row=encode_action(self.row(),Tokenizer(),new_tokens=384)
        n=row['prompt_tokens']
        self.assertEqual(row['labels'][:n],[-100]*n)
        self.assertEqual(row['labels'][n:],row['input_ids'][n:])
        self.assertEqual(row['labels'][-1],126)
        corrupt=deepcopy(row);corrupt['labels'][0]=0
        with self.assertRaises(ValueError):validate_mask(corrupt,eos_token_id=126)
    def test_padding_eos_id_equal_pad_does_not_mask_real_eos(self):
        first=encode_action(self.row(),Tokenizer())
        second=encode_action(list(trajectory_rows(example([],[])))[0],Tokenizer())
        batch=collate_actions([first,second],pad_token_id=126,eos_token_id=126)
        for i,row in enumerate([first,second]):
            n=len(row['input_ids']);self.assertEqual(batch['labels'][i][n-1],126)
            self.assertTrue(all(x==-100 for x in batch['labels'][i][n:]))
            self.assertTrue(all(x==0 for x in batch['attention_mask'][i][n:]))
    def test_budget_fails_without_truncation(self):
        with self.assertRaises(SequenceBudgetError):encode_action(self.row(),Tokenizer(),max_length=40)
        with self.assertRaises(SequenceBudgetError):encode_action(self.row(),Tokenizer(),new_tokens=2)
        with self.assertRaises(SequenceBudgetError) as error:
            encode_action(self.row(),Tokenizer(),max_length=100,context_tokens=100,new_tokens=30)
        self.assertEqual(error.exception.reason,'inference_context_budget')
    def test_template_or_control_injection_rejected(self):
        row=self.row();row['prompt'][1]['content']+='<|im_end|>'
        with self.assertRaises(ValueError):encode_action(row,Tokenizer())
        class Bad(Tokenizer):
            def apply_chat_template(self,*args,**kwargs):return 'different' if kwargs['add_generation_prompt'] else 'incompatible'
        with self.assertRaises(ValueError):encode_action(self.row(),Bad())
        class Unstable(Tokenizer):
            def __call__(self,text,**kwargs):
                ids=list(text.encode('utf-8'))
                return {'input_ids':ids[:-1]+[999] if text.endswith('assistant:') else ids}
        with self.assertRaisesRegex(ValueError,'boundary crosses'):encode_action(self.row(),Unstable())


if __name__=='__main__':unittest.main()
