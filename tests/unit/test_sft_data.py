import json
from pathlib import Path
import unittest

import tests.integration.test_trajectory_dataset as fixtures
from training.sft_data import TokenRows,select_rows,verify_dataset
from lifemate.observability.provenance import file_hash


class SFTDataTests(unittest.TestCase):
    def fixture(self):
        case=fixtures.DatasetTests('test_build_audit_and_no_dev_or_final_state');case.setUp()
        self.addCleanup(case.doCleanups);case.build()
        path=case.out/'dataset_manifest.json';m=json.loads(path.read_text(encoding='utf-8'))
        m.update({'protocol':'custom_json','policy':'off'})
        path.write_text(json.dumps(m),encoding='utf-8')
        (case.out/'audit.json').write_text(json.dumps({'valid':True,'dataset_manifest_sha256':file_hash(path)}),encoding='utf-8')
        return case

    def test_verified_index_and_deterministic_longest_selection(self):
        case=self.fixture();_,split,_=verify_dataset(case.out)
        rows=TokenRows(case.out/'train.tokens.jsonl',partition='train',allowed_ids=set(split['train_ids']),eos_token_id=126)
        selected=select_rows(rows,3,longest_first=True)
        self.assertEqual(selected,select_rows(rows,3,longest_first=True))
        self.assertEqual(rows.entries[selected[0]]['sequence_tokens'],max(e['sequence_tokens'] for e in rows.entries))
        self.assertTrue(all(rows[i]['sample_id'] in split['train_ids'] for i in selected))
        self.assertEqual(len(select_rows(rows,0)),len(rows))

    def test_mutation_and_stale_audit_rejected(self):
        case=self.fixture();path=case.out/'train.tokens.jsonl'
        path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
        with self.assertRaises(ValueError):verify_dataset(case.out)
        case=self.fixture();audit=case.out/'audit.json'
        audit.write_text(json.dumps({'valid':True,'dataset_manifest_sha256':'wrong'}),encoding='utf-8')
        with self.assertRaises(ValueError):verify_dataset(case.out)

    def test_wrong_partition_or_short_length_not_silently_filtered(self):
        case=self.fixture()
        with self.assertRaises(ValueError):TokenRows(case.out/'train.tokens.jsonl',partition='validation',allowed_ids={'z'},eos_token_id=126)
        split=json.loads((case.out/'inner_split.json').read_text(encoding='utf-8'))
        with self.assertRaises(ValueError):TokenRows(case.out/'train.tokens.jsonl',partition='train',allowed_ids=set(split['train_ids']),eos_token_id=126,max_length=10)


if __name__=='__main__':unittest.main()
