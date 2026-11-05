"""Independent artifact/prefix/token/label audit; no LLM training or ToolEnv."""
import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path

from lifemate.agent.planner import parse_action
from lifemate.agent.prompt import messages, SYSTEM
from lifemate.input.adapter import _parse
from lifemate.observability.provenance import file_hash
from training.prepare_trajectories import safe_path,write_json
from training.loss_mask import torch_mask_probe


def audit_dataset(directory,tokenizer,*,torch_probe=False):
    directory=safe_path(directory)
    manifest=_parse((directory/'dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete':raise ValueError('dataset not complete')
    expected={'inner_split.json','report.json','quarantine.jsonl','train.jsonl','validation.jsonl',
              'train.tokens.jsonl','validation.tokens.jsonl'}
    if set(manifest['files_sha256'])!=expected:raise ValueError('unexpected dataset files')
    for name,digest in manifest['files_sha256'].items():
        if file_hash(directory/name)!=digest:raise ValueError('dataset file fingerprint mismatch')
    split=_parse((directory/'inner_split.json').read_text(encoding='utf-8'))
    report=_parse((directory/'report.json').read_text(encoding='utf-8'))
    sets={p:set(split[key]) for p,key in [('train','train_ids'),('validation','validation_ids')]}
    if sets['train']&sets['validation'] or (sets['train']|sets['validation'])&set(split['outer_dev_ids']):
        raise ValueError('dataset split leakage')
    if set(split['seen_ids'])&sets['validation']:raise ValueError('seen sample in validation')
    for group in split['groups']:
        if set(group)&sets['train'] and set(group)&sets['validation']:raise ValueError('inner group leakage')
    rows_total=0;supervised=0;ignored=0;batch=[];retained={p:set() for p in sets};seen_row_ids=set()
    counts={p:0 for p in sets}
    for partition in sets:
        previous=None;sample_steps={}
        with (directory/(partition+'.jsonl')).open(encoding='utf-8') as raw, (directory/(partition+'.tokens.jsonl')).open(encoding='utf-8') as token:
            for left,right in zip_longest(raw,token):
                if left is None or right is None:raise ValueError('raw/token rows differ')
                row,encoded=_parse(left),_parse(right);sid=row['sample_id'];step=row['step_index']
                if sid not in sets[partition] or row['partition']!=partition or encoded['partition']!=partition:
                    raise ValueError('wrong row partition')
                if row['id'] in seen_row_ids or row['id']!=f'{sid}:{step}':raise ValueError('duplicate/invalid row ID')
                seen_row_ids.add(row['id'])
                if any(encoded[k]!=row[k] for k in ('id','sample_id','step_index')):raise ValueError('row identity differs')
                if step!=sample_steps.get(sid,0):raise ValueError('missing/out-of-order trajectory step')
                sample_steps[sid]=step+1
                if type(row['trajectory_rows']) is not int or type(step) is not int or step>=row['trajectory_rows']:raise ValueError('invalid trajectory size')
                if len(row['prompt'])!=2 or len(row['completion'])!=1:raise ValueError('unexpected training message structure')
                if row['completion'][0]['role']!='assistant':raise ValueError('target role differs')
                target=row['completion'][0]['content'];action=parse_action(target)
                if action['type']!=row['target_type'] or action.get('name')!=row['target_tool']:raise ValueError('target metadata differs')
                body=_parse(row['prompt'][1]['content'])
                if set(body)!={'input','current_turn_events'} or set(body['input'])!={'request','context','history','tools'}:
                    raise ValueError('unexpected input/reference fields')
                events=body['current_turn_events']
                if len(events)!=step or row['prompt']!=messages(body['input'],events):raise ValueError('prefix/current inference messages differ')
                if action['type']=='call' and action['name'] not in {t['name'] for t in body['input']['tools']}:
                    raise ValueError('target tool not in allowlist')
                if step:
                    if previous is None or previous['sid']!=sid or previous['action']['type']!='call':raise ValueError('noncontiguous/terminal prefix')
                    if row['trajectory_rows']!=previous['expected']:raise ValueError('trajectory size changed')
                    if body['input']!=previous['input'] or events[:-1]!=previous['events'] or events[-1]['action']!=previous['action']:
                        raise ValueError('prefix contains changed input or wrong prior action')
                    if previous['action']['name']=='sys.ask_missing_info':raise ValueError('row after terminal ask')
                    if set(events[-1])!={'action','observation'}:raise ValueError('invalid prefix event fields')
                elif previous and previous['steps']!=previous['expected']:raise ValueError('incomplete previous trajectory')
                if step==row['trajectory_rows']-1 and not (action['type']=='finish' or action.get('name')=='sys.ask_missing_info'):
                    raise ValueError('trajectory lacks terminal target')
                if step<row['trajectory_rows']-1 and action['type']=='finish':raise ValueError('premature finish target')
                prefix=tokenizer.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True)
                prefix_ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
                ids=encoded['input_ids'];labels=encoded['labels'];n=len(prefix_ids)
                full_ids=tokenizer(prefix+target+tokenizer.eos_token,add_special_tokens=False)['input_ids']
                if ids!=full_ids or encoded['prompt_tokens']!=n or encoded['target_tokens']!=len(ids)-n:
                    raise ValueError('actual tokenizer sequence/boundary differs')
                if ids[:n]!=prefix_ids or labels!=[-100]*n+ids[n:]:raise ValueError('prompt/completion labels wrong')
                if len(ids)!=len(labels) or encoded['attention_mask']!=[1]*len(ids):raise ValueError('invalid mask dimensions')
                if ids[-1]!=tokenizer.eos_token_id or labels[-1]!=tokenizer.eos_token_id:raise ValueError('untrained EOS')
                if tokenizer.decode(ids[n:-1],skip_special_tokens=False)!=target:raise ValueError('trained target differs')
                if len(ids)>report['budgets']['training_max_length'] or n+384>16384 or len(ids)-n>384:
                    raise ValueError('unsupported inference/training budget')
                if len(batch)<4:batch.append(encoded)
                supervised+=len(ids)-n;ignored+=n;rows_total+=1;counts[partition]+=1;retained[partition].add(sid)
                previous={'sid':sid,'action':action,'input':body['input'],'events':events,
                          'steps':step+1,'expected':row['trajectory_rows']}
                if rows_total%500==0:print(f'Audited {rows_total} rows',flush=True)
        if previous and previous['steps']!=previous['expected']:raise ValueError('incomplete last trajectory')
        if counts[partition]!=report['partitions'][partition]['rows'] or len(retained[partition])!=report['partitions'][partition]['retained_samples']:
            raise ValueError('retained counts differ from report')
    quarantined={p:set() for p in sets}
    with (directory/'quarantine.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=_parse(line);p=row['partition'];sid=row['sample_id']
            if sid not in sets[p] or sid in quarantined[p] or sid in retained[p]:raise ValueError('quarantine identity mismatch')
            quarantined[p].add(sid)
    for p in sets:
        if retained[p]|quarantined[p]!=sets[p]:raise ValueError('reserved sample missing from exports/quarantine')
    if rows_total!=report['mask_checks_passed']:raise ValueError('mask audit count differs')
    # Verify artifacts did not change while the audit streamed them.
    for name,digest in manifest['files_sha256'].items():
        if file_hash(directory/name)!=digest:raise ValueError('dataset changed during audit')
    return {'schema_version':1,'valid':True,'rows':rows_total,'counts':counts,
            'supervised_tokens':supervised,'ignored_prompt_tokens':ignored,
            'mask_semantics':'Target JSON and EOS only; assistant header and padding excluded. Standard causal shift.',
            'gold_prefix_authenticity':'Structural prefix/fingerprints checked; observations are not independently replayed here.',
            'torch_probe':torch_mask_probe(batch,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id) if torch_probe else None,
            'training_performed':False,'dataset_manifest_sha256':file_hash(directory/'dataset_manifest.json')}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--torch-probe',action='store_true')
    a=p.parse_args(argv)
    directory,output,model=safe_path(a.dataset_dir),safe_path(a.output),safe_path(a.model_path)
    if output.exists() or output.parent!=directory:raise ValueError('choose a new audit filename inside dataset-dir')
    from transformers import AutoTokenizer
    from training.qwen35 import patch_tokenizer_loading
    patch_tokenizer_loading()  # no-op unless the tokenizer config is Qwen3.5
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    result=audit_dataset(directory,tokenizer,torch_probe=a.torch_probe)
    write_json(output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
