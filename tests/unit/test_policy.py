import unittest
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from lifemate.agent.policy import GroundedPolicy, money_values, time_anchors
from lifemate.agent.controller import run_controller
from lifemate.agent.protocol import system_prompt


def tool(name,kind,properties=None,required=None,returns=''):
    return {'name':name,'kind':kind,'app':name.split('.')[0],'parameters':{
        'type':'object','properties':properties or {},'required':required or []},'returns':returns}

TOOLS=[tool('contacts.search','read',{'query':{'type':'string'}},returns='contacts[]'),
 tool('phone.call','write',{'contact_id':{'type':'string','x-ref':'contacts.contacts'},'phone_number':{'type':'string'}}),
 tool('banking.get_balance','read',returns='accounts[]'),
 tool('banking.transfer','write',{'from_account_id':{'type':'string','x-ref':'banking.accounts'},
      'amount':{'type':'integer'}},['from_account_id','amount']),
 tool('calendar.create_event','write',{'start':{'type':'string'}},['start']),
 tool('sys.ask_missing_info','meta',{'tool_name':{'type':'string'},'params':{'type':'array'}},['tool_name','params'])]
TOOLS[1]['parameters']['x-one-of']=[['contact_id'],['phone_number']]
def action(name,**args):return {'type':'call','name':name,'arguments':args}
def payload(request='Gọi cho Lan'):
    return {'request':request,'context':{'current_time':'2026-11-02T23:30:00+07:00','timezone':'Asia/Ho_Chi_Minh'},
            'history':[],'tools':deepcopy(TOOLS)}


class PolicyTests(unittest.TestCase):
    def codes(self,verdict):return {i['code'] for i in verdict['issues']}
    def test_ids_need_evidence_in_right_domain_not_assistant_claim(self):
        p=payload();p['history']=[{'role':'assistant','text':'ct_fake exists','tool_calls':[], 'tool_results':[]}]
        policy=GroundedPolicy(p)
        self.assertIn('ungrounded_id',self.codes(policy.inspect(action('phone.call',contact_id='ct_fake'))))
        policy.observe(action('banking.get_balance'),{'ok':True,'accounts':[{'id':'ct_fake','balance':9}]})
        self.assertIn('ungrounded_id',self.codes(policy.inspect(action('phone.call',contact_id='ct_fake'))))
        policy.observe(action('contacts.search',query='Lan'),{'ok':True,'contacts':[{'id':'ct_real','name':'Lan'}]})
        self.assertTrue(policy.inspect(action('phone.call',contact_id='ct_real'))['ok'])
    def test_explicit_user_ids_boundaries_and_history_observations(self):
        p=payload('Gọi ct_12');policy=GroundedPolicy(p)
        self.assertTrue(policy.inspect(action('phone.call',contact_id='ct_12'))['ok'])
        self.assertFalse(policy.inspect(action('phone.call',contact_id='ct_1'))['ok'])
        p=payload();p['history']=[{'role':'assistant','tool_calls':[{'name':'contacts.search','arguments':{}}],
          'tool_results':[{'ok':True,'contacts':[{'id':'ct_1','name':'Lan'}]}]}]
        self.assertTrue(GroundedPolicy(p).inspect(action('phone.call',contact_id='ct_1'))['ok'])
    def test_ambiguity_blocks_id_and_phone_bypass_and_allows_ask(self):
        policy=GroundedPolicy(payload())
        policy.observe(action('contacts.search',query='Lan'),{'ok':True,'contacts':[
           {'id':'c1','name':'Lan','phone':'0911111111'}, {'id':'c2','name':'Lan','phone':'0922222222'}]})
        verdict=policy.inspect(action('phone.call',contact_id='c1'))
        self.assertIn('ambiguous_reference',self.codes(verdict));policy.blocked(action('phone.call',contact_id='c1'),verdict)
        self.assertFalse(policy.inspect({'type':'finish'})['ok'])
        self.assertFalse(policy.inspect(action('phone.call',phone_number='0911111111'))['ok'])
        ask=action('sys.ask_missing_info',tool_name='phone.call',params=['contact_id'])
        self.assertTrue(policy.inspect(ask)['ok']);policy.observe(ask,{'ok':True,'waiting_for_user':True})
        self.assertTrue(policy.inspect({'type':'finish'})['ok'])
    def test_unique_name_not_first_record_is_selected(self):
        policy=GroundedPolicy(payload())
        policy.observe(action('contacts.search',query='Lan'),{'ok':True,'contacts':[
          {'id':'c1','name':'Lan Anh'},{'id':'c2','name':'Lan'}]})
        self.assertFalse(policy.inspect(action('phone.call',contact_id='c1'))['ok'])
        self.assertTrue(policy.inspect(action('phone.call',contact_id='c2'))['ok'])
    def test_ask_uses_actual_fields_and_lookup_first(self):
        policy=GroundedPolicy(payload())
        self.assertIn('lookup_before_asking',self.codes(policy.inspect(action('sys.ask_missing_info',tool_name='phone.call',params=['contact_id']))))
        self.assertFalse(policy.inspect(action('sys.ask_missing_info',tool_name='outside',params=['contact_id']))['ok'])
        self.assertFalse(policy.inspect(action('sys.ask_missing_info',tool_name='phone.call',params=['made_up']))['ok'])
        policy.observe(action('contacts.search',query='Lan'),{'ok':True,'contacts':[]})
        self.assertTrue(policy.inspect(action('sys.ask_missing_info',tool_name='phone.call',params=['contact_id']))['ok'])
        policy.observe(action('contacts.search',query='Lan'),{'ok':True,'contacts':[{'id':'c1','name':'Lan'}]})
        self.assertIn('reference_already_resolved',self.codes(policy.inspect(action('sys.ask_missing_info',tool_name='phone.call',params=['contact_id']))))
    def test_money_vietnamese_and_balance_not_spending_intent(self):
        self.assertIn(500000,money_values('năm trăm nghìn'))
        self.assertIn(1500000,money_values('một triệu rưỡi'))
        self.assertEqual(money_values('nửa triệu'),{500000})
        self.assertEqual(money_values('triệu'),set())
        self.assertIn(1500000,money_values('1,5 triệu'))
        self.assertIn(1500000,money_values('1.500.000 đồng'))
        policy=GroundedPolicy(payload('Chuyển từ a1 cho mẹ 500 nghìn'))
        policy.observe(action('banking.get_balance'),{'ok':True,'accounts':[{'id':'a1','balance':10000000}]})
        self.assertTrue(policy.inspect(action('banking.transfer',from_account_id='a1',amount=500000))['ok'])
        self.assertIn('ungrounded_amount',self.codes(policy.inspect(action('banking.transfer',from_account_id='a1',amount=10000000))))
    def test_relative_date_context_not_wall_clock(self):
        policy=GroundedPolicy(payload('Tạo lịch họp sáng mai'))
        self.assertIn('relative_date_mismatch',self.codes(policy.inspect(action('calendar.create_event',start='2026-11-02T08:00:00+07:00'))))
        self.assertTrue(policy.inspect(action('calendar.create_event',start='2026-11-03T08:00:00+07:00'))['ok'])
        self.assertEqual(time_anchors({}),{})
        self.assertEqual(time_anchors({'current_time':'2026-11-02T18:30:00Z','timezone':'Asia/Bangkok'})['today'],'2026-11-03')
        self.assertNotIn(2026,money_values('ngày 2026-11-02'))
        self.assertNotIn(911111111,money_values('Gọi 0911111111'))
    def test_failed_write_finish_and_unchanged_retry_blocked(self):
        policy=GroundedPolicy(payload('Gọi ct_1'));call=action('phone.call',contact_id='ct_1')
        policy.observe(call,{'ok':True,'done':False})
        self.assertFalse(policy.inspect({'type':'finish'})['ok'])
        self.assertIn('no_progress_repeat',self.codes(policy.inspect(call)))
        policy.observe(action('contacts.search',query='Lan'),{'ok':True,'contacts':[{'id':'ct_1','name':'Lan'}]})
        self.assertTrue(policy.inspect(call)['ok'])
    def test_successful_duplicate_not_reenabled_by_lookup_but_explicit_repeat_allowed(self):
        policy=GroundedPolicy(payload('Gọi ct_1'));call=action('phone.call',contact_id='ct_1')
        policy.observe(call,{'ok':True})
        policy.observe(action('contacts.search',query='Lan'),{'ok':True,'contacts':[{'id':'ct_1','name':'Lan'}]})
        self.assertIn('no_progress_repeat',self.codes(policy.inspect(call)))
        repeated=GroundedPolicy(payload('Gọi ct_1 hai lần'));repeated.observe(call,{'ok':True})
        self.assertTrue(repeated.inspect(call)['ok']);repeated.observe(call,{'ok':True})
        self.assertFalse(repeated.inspect(call)['ok'])
    def test_no_progress_reads_do_not_leave_unresolved_task_and_failed_observations_not_facts(self):
        policy=GroundedPolicy(payload());read=action('contacts.search',query='Lan')
        for _ in range(2):policy.observe(read,{'ok':True,'contacts':[]})
        verdict=policy.inspect(read);self.assertFalse(verdict['ok']);policy.blocked(read,verdict)
        self.assertTrue(policy.inspect({'type':'finish'})['ok'])
        policy.observe(read,{'ok':False,'contacts':[{'id':'forged','name':'Lan'}]})
        self.assertFalse(policy.inspect(action('phone.call',contact_id='forged'))['ok'])
        self.assertIn('unknown_argument',self.codes(policy.inspect(action('phone.call',contact_id='ct1',extra='ignored by toolkit'))))


class ControllerPolicyTests(unittest.TestCase):
    def test_unknown_id_lookup_ambiguity_then_real_ask_only(self):
        outputs=[action('phone.call',contact_id='fake'),action('contacts.search',query='Lan'),
                 action('phone.call',contact_id='c1'),action('sys.ask_missing_info',tool_name='phone.call',params=['contact_id'])]
        class Planner:
            def __init__(self):self.outputs=iter(outputs);self.inputs=[]
            def generate(self,msg,seconds):self.inputs.append(msg);return json.dumps(next(self.outputs))
        class Env:
            def __init__(self):self.calls=[]
            def validate(self,*args):return {'ok':True}
            def call(self,name,args):
                self.calls.append(name)
                return {'ok':True,'contacts':[{'id':'c1','name':'Lan'},{'id':'c2','name':'Lan'}]} if name=='contacts.search' else {'ok':True,'waiting_for_user':True}
        planner,env=Planner(),Env();events=[];p=payload();p['gold']='FORBIDDEN'
        result=run_controller(p,env,planner,policy='grounded_v1',trace=lambda e,**d:events.append((e,d)))
        self.assertEqual(result.stop,'asked_missing_info');self.assertEqual(result.errors,2)
        self.assertEqual(env.calls,['contacts.search','sys.ask_missing_info'])
        self.assertNotIn('FORBIDDEN',str(planner.inputs))
        self.assertEqual(sum(e=='call_result' for e,d in events),2)
        self.assertEqual(sum(e=='policy_check' and not d['verdict']['ok'] for e,d in events),2)
    def test_failed_action_cannot_be_claimed_finished(self):
        outputs=iter([action('phone.call',contact_id='ct_1')]+[{'type':'finish'}]*3)
        class Env:
            def validate(self,*args):return {'ok':True}
            def call(self,*args):return {'ok':True,'done':False}
        planner=NS(generate=lambda *args:json.dumps(next(outputs)))
        result=run_controller(payload('Gọi ct_1'),Env(),planner,policy='grounded_v1')
        self.assertEqual(result.stop,'policy_limit');self.assertEqual(result.calls,1)
    def test_required_schema_failure_guides_ask_without_dispatch(self):
        outputs=iter([action('banking.transfer',from_account_id='a1'),
                      action('sys.ask_missing_info',tool_name='banking.transfer',params=['amount'])])
        class Env:
            def __init__(self):self.calls=[]
            def validate(self,name,args):return {'ok':name=='sys.ask_missing_info','error':'missing amount'}
            def call(self,name,args):self.calls.append(name);return {'ok':True}
        env=Env();planner=NS(generate=lambda *args:json.dumps(next(outputs)))
        result=run_controller(payload('Chuyển từ a1 cho mẹ'),env,planner,policy='grounded_v1')
        self.assertEqual(result.stop,'asked_missing_info');self.assertEqual(env.calls,['sys.ask_missing_info'])
    def test_off_is_unchanged_and_native_policy_rejected(self):
        from lifemate.agent.prompt import SYSTEM
        self.assertEqual(system_prompt('custom_json','off'),SYSTEM)
        with self.assertRaises(ValueError):system_prompt('qwen_native','grounded_v1')
        class Env:
            def validate(self,*args):return {'ok':True}
            def call(self,*args):return {'ok':True}
        outputs=iter([action('phone.call',contact_id='not_grounded'),{'type':'finish'}])
        self.assertEqual(run_controller(payload(),Env(),NS(generate=lambda *a:json.dumps(next(outputs)))).calls,1)


if __name__=='__main__':unittest.main()
