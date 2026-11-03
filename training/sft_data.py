"""Seek-indexed audited token rows: no truncation, re-tokenization or tool access."""
import json
from pathlib import Path
import random

from lifemate.input.adapter import _parse
from lifemate.observability.provenance import file_hash
from training.loss_mask import validate_mask
from training.prepare_trajectories import safe_path

FILES={'inner_split.json','report.json','quarantine.jsonl','train.jsonl','validation.jsonl',
       'train.tokens.jsonl','validation.tokens.jsonl'}


def verify_dataset(directory,audit_file=None):
    directory=safe_path(directory)
    manifest=_parse((directory/'dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete' or manifest.get('protocol')!='custom_json' or manifest.get('policy')!='off':
        raise ValueError('unsupported/incomplete SFT dataset')
    if set(manifest['files_sha256'])!=FILES:raise ValueError('unexpected dataset file set')
    for name,digest in manifest['files_sha256'].items():
        if file_hash(directory/name)!=digest:raise ValueError('dataset fingerprint mismatch')
    audit_file=safe_path(audit_file or directory/'audit.json')
    audit=_parse(audit_file.read_text(encoding='utf-8'))
    if not audit.get('valid') or audit.get('dataset_manifest_sha256')!=file_hash(directory/'dataset_manifest.json'):
        raise ValueError('fresh successful mask audit required')
    split=_parse((directory/'inner_split.json').read_text(encoding='utf-8'))
    train,validation,dev=map(set,(split['train_ids'],split['validation_ids'],split['outer_dev_ids']))
    if train&validation or (train|validation)&dev or set(split['seen_ids'])&validation:
        raise ValueError('invalid SFT partitions')
    report=_parse((directory/'report.json').read_text(encoding='utf-8'))
    return manifest,split,report


class TokenRows:
    def __init__(self,path,*,partition,allowed_ids,eos_token_id,max_length=16384):
        self.path=safe_path(path);self.partition=partition;self.eos_token_id=eos_token_id
        self.entries=[];seen=set();offset=0
        with self.path.open('rb') as stream:
            for line in stream:
                row=_parse(line.decode('utf-8'))
                if row['id'] in seen or row['sample_id'] not in allowed_ids or row.get('partition')!=partition:
                    raise ValueError('token row identity/partition mismatch')
                validate_mask(row,eos_token_id=eos_token_id)
                if len(row['input_ids'])>max_length:raise ValueError('training length too small; regenerate explicitly, never truncate')
                self.entries.append({'offset':offset,'bytes':len(line),'id':row['id'],'sample_id':row['sample_id'],
                    'sequence_tokens':len(row['input_ids']),'target_tokens':row['target_tokens']})
                offset+=len(line);seen.add(row['id'])
        if not self.entries:raise ValueError('empty token partition')

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        entry=self.entries[index]
        with self.path.open('rb') as stream:
            stream.seek(entry['offset']);row=_parse(stream.read(entry['bytes']).decode('utf-8'))
        if row['id']!=entry['id']:raise ValueError('token source changed after indexing')
        validate_mask(row,eos_token_id=self.eos_token_id)
        return row


def select_rows(dataset,limit,*,seed=42,longest_first=False):
    if type(limit) is not int or limit<0:raise ValueError('row limit must be nonnegative; zero selects all')
    indices=list(range(len(dataset)));rng=random.Random(seed)
    rng.shuffle(indices)
    if longest_first:
        longest=max(range(len(dataset)),key=lambda i:dataset.entries[i]['sequence_tokens'])
        indices.remove(longest);indices.insert(0,longest)
    return indices[:limit] if limit else indices
