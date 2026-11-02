from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from lifemate.observability.provenance import file_hash
from training.prepare_trajectories import build_dataset,write_json
from training.audit_trajectories import audit_dataset
from tests.unit.test_trajectory import Tokenizer


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.data=self.root/'Data';train=self.data/'train';train.mkdir(parents=True)
        self.tool={'name':'notes.create','app':'notes','kind':'write','parameters':{
            'type':'object','properties':{'title':{'type':'string'}},'required':['title']}}
        catalog={'schema_version':'test','apps':[{'id':'notes','tools':[self.tool]},
            {'id':'sys','tools':[{'name':'sys.ask_missing_info','app':'sys','parameters':{'type':'object'}}]}]}
        self.rows=[]
        for sid in ['a','b','c','d','e','z']:
            self.rows.append({'id':sid,'audio':'audio/a.wav','audio_text':'synthetic request',
                'context':{'current_time':'2026-11-02T10:00:00+07:00','timezone':'Asia/Bangkok',
                    'location':{},'user_profile':{},'installed_apps':{'notes':{'tools':['notes.create']}}},'history':[],
                'label':{'tool_calls':[{'name':'notes.create','arguments':{'title':'target'}}],
                         'tool_results':[{'ok':True}],'final_state':{'NEVER_EXPORT':'PRIVATE_STATE'}}})
        # A broken held-out Reference is deliberately ignored by the build.
        self.rows[-1]['label']=None
        (self.data/'VERSION').write_text('fixture',encoding='utf-8')
        (train/'train_tools_catalog.json').write_text(json.dumps(catalog),encoding='utf-8')
        self.out=self.root/'output';self.refresh()

    def refresh(self):
        (self.data/'train/train.jsonl').write_text('\n'.join(json.dumps(r) for r in self.rows)+'\n',encoding='utf-8')
        self.manifest={'train_ids':['a','b','c','d','e'],'dev_ids':['z'],
                       'groups':[['a','b'],['c'],['d'],['e'],['z']],
                       'source_sha256':{name:file_hash(self.data/name) for name in
                            ['VERSION','train/train.jsonl','train/train_tools_catalog.json']}}

    def build(self,**kwargs):
        with redirect_stdout(io.StringIO()):
            report=build_dataset(self.data,self.manifest,Tokenizer(),self.out,seen_ids=['a'],validation_fraction=.2,**kwargs)
        write_json(self.out/'report.json',report)
        write_json(self.out/'dataset_manifest.json',{'status':'complete',
            'files_sha256':{p.name:file_hash(p) for p in self.out.iterdir() if p.is_file()}})
        return report

    def rehash(self,name):
        path=self.out/'dataset_manifest.json';m=json.loads(path.read_text(encoding='utf-8'))
        m['files_sha256'][name]=file_hash(self.out/name);path.write_text(json.dumps(m),encoding='utf-8')

    def test_build_audit_and_no_dev_or_final_state(self):
        report=self.build();audit=audit_dataset(self.out,Tokenizer())
        self.assertTrue(audit['valid']);self.assertEqual(audit['rows'],10)
        self.assertEqual(report['outer_dev_samples_excluded'],1)
        self.assertNotIn('PRIVATE_STATE',(self.out/'train.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(report['inner_reserved']['forced_train_ids'],2)
        with self.assertRaises(FileExistsError):self.build()

    def test_bad_source_manifest_rejected_before_output(self):
        self.manifest['source_sha256']['VERSION']='wrong'
        with self.assertRaises(ValueError):self.build()
        self.assertFalse(self.out.exists())

    def test_quarantine_entire_trajectory_without_truncation(self):
        self.rows[0]['label']['tool_calls'][0]['arguments']['title']='x'*1000;self.refresh()
        report=self.build()
        self.assertEqual(report['partitions']['train']['quarantined_samples'],1)
        self.assertEqual(report['partitions']['train']['quarantined_rows'],2)
        retained=[json.loads(x)['sample_id'] for x in (self.out/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertNotIn('a',retained);self.assertTrue(audit_dataset(self.out,Tokenizer())['valid'])

    def test_error_policy_leaves_no_completed_manifest(self):
        self.rows[0]['label']['tool_calls'][0]['arguments']['title']='x'*1000;self.refresh()
        with self.assertRaises(ValueError):self.build(overflow_policy='error')
        self.assertFalse((self.out/'dataset_manifest.json').exists())

    def test_independent_audit_finds_mask_error_even_after_rehash(self):
        self.build();path=self.out/'train.tokens.jsonl'
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['labels'][0]=rows[0]['input_ids'][0]
        path.write_text('\n'.join(map(json.dumps,rows))+'\n',encoding='utf-8');self.rehash(path.name)
        with self.assertRaisesRegex(ValueError,'labels wrong'):audit_dataset(self.out,Tokenizer())

    def test_independent_audit_finds_future_event_even_after_rehash(self):
        self.build();path=self.out/'train.jsonl'
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        content=json.loads(rows[0]['prompt'][1]['content']);content['current_turn_events'].append({'future':'FORBIDDEN'})
        rows[0]['prompt'][1]['content']=json.dumps(content)
        path.write_text('\n'.join(map(json.dumps,rows))+'\n',encoding='utf-8');self.rehash(path.name)
        with self.assertRaisesRegex(ValueError,'prefix/current inference'):audit_dataset(self.out,Tokenizer())


if __name__=='__main__':unittest.main()
