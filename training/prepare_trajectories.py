"""Build custom-JSON SFT examples and explicit completion-only labels; no training."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import time

from lifemate.agent.prompt import SYSTEM
from lifemate.input.adapter import DatasetLoader, _parse
from lifemate.observability.provenance import collect_provenance, file_hash
from training.trajectories import explicit_inner_split, inner_split, trajectory_rows
from training.loss_mask import encode_action, SequenceBudgetError, collate_actions


def safe_path(path):
    path=Path(path).resolve()
    if any(p.lower().endswith('_environment') for p in path.parts):
        raise ValueError('environment paths are forbidden')
    return path


def verify_sources(root,manifest):
    fingerprints=manifest['source_sha256']
    if not {'VERSION','train/train.jsonl','train/train_tools_catalog.json'}<=set(fingerprints):
        raise ValueError('manifest lacks required source fingerprints')
    for name,digest in fingerprints.items():
        source=safe_path(root/name)
        if not source.is_relative_to(root) or file_hash(source)!=digest:
            raise ValueError('manifest source fingerprint/path mismatch')


def write_json(path,data):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n')


def build_dataset(root,manifest,tokenizer,out,*,seen_ids=(),validation_fraction=.1,seed=43,
                  max_length=16384,overflow_policy='quarantine',explicit_split=None):
    """Publish completion marker last. On failure retain partial diagnostics only."""
    root,out=safe_path(root),safe_path(out)
    if out.is_relative_to(root):raise ValueError('output must be outside Data')
    if out.exists():raise FileExistsError('choose a new dataset output directory')
    if overflow_policy not in {'quarantine','error'}:raise ValueError('invalid overflow policy')
    if type(max_length) is not int or not 0<max_length<=16384:raise ValueError('invalid max length')
    verify_sources(root,manifest)
    split=(explicit_inner_split(manifest,explicit_split) if explicit_split is not None else
           inner_split(manifest,validation_fraction=validation_fraction,seed=seed,seen_ids=seen_ids))
    loader=DatasetLoader(root,'train',check_audio=False)
    observed={item.id for item in loader.iter_inputs()}
    if observed!=set(manifest['train_ids'])|set(manifest['dev_ids']):raise ValueError('manifest/metadata ID coverage differs')
    memberships={sid:partition for partition,key in [('train','train_ids'),('validation','validation_ids')] for sid in split[key]}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'inner_split.json',split)
    totals={p:Counter() for p in ('train','validation')};tools={p:Counter() for p in totals}
    lengths={p:{'max_prompt':0,'max_target':0,'max_sequence':0,'sum_prompt':0,'sum_target':0} for p in totals}
    rejected=Counter();mask_checks=0;started=time.perf_counter();sample_batch=[];processed=0
    with ExitStack() as stack:
        raw={p:stack.enter_context((out/(p+'.jsonl')).open('x',encoding='utf-8',newline='\n')) for p in totals}
        encoded={p:stack.enter_context((out/(p+'.tokens.jsonl')).open('x',encoding='utf-8',newline='\n')) for p in totals}
        quarantine=stack.enter_context((out/'quarantine.jsonl').open('x',encoding='utf-8',newline='\n'))
        for example in loader.iter_training(sample_ids=manifest['train_ids']):
            sid=example.agent_input.id;partition=memberships[sid]
            rows=list(trajectory_rows(example));tokens=[];failure=None
            if len(example.reference.tool_calls)>8 or len(rows)>12:
                failure={'reason':'controller_call_or_decision_budget','step_index':None}
            else:
                for row in rows:
                    try:tokens.append(encode_action(row,tokenizer,max_length=max_length))
                    except SequenceBudgetError as error:
                        failure={'reason':error.reason,'step_index':row['step_index'],
                                 'prompt_tokens':error.prompt_tokens,'target_tokens':error.target_tokens}
                        break
            if failure:
                rejected[failure['reason']]+=1
                quarantine.write(json.dumps({'sample_id':sid,'partition':partition,
                    'withheld_trajectory_rows':len(rows),**failure},ensure_ascii=False)+'\n')
                if overflow_policy=='error':raise ValueError('trajectory exceeds configured budget; no completed dataset')
                totals[partition]['quarantined_samples']+=1
                totals[partition]['quarantined_rows']+=len(rows)
            else:
                totals[partition]['retained_samples']+=1
                totals[partition]['samples_with_history']+=bool(example.agent_input.history)
                totals[partition]['out_of_scope_samples']+=not example.reference.tool_calls
                totals[partition]['ask_samples']+=any(c['name']=='sys.ask_missing_info' for c in example.reference.tool_calls)
                for row,tokenized in zip(rows,tokens):
                    row['partition']=partition;tokenized['partition']=partition
                    raw[partition].write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                    encoded[partition].write(json.dumps(tokenized,ensure_ascii=False,separators=(',',':'))+'\n')
                    totals[partition]['rows']+=1;totals[partition][row['target_type']+'_targets']+=1
                    if row['target_tool']:tools[partition][row['target_tool']]+=1
                    stats=lengths[partition]
                    stats['max_prompt']=max(stats['max_prompt'],tokenized['prompt_tokens'])
                    stats['max_target']=max(stats['max_target'],tokenized['target_tokens'])
                    stats['max_sequence']=max(stats['max_sequence'],len(tokenized['input_ids']))
                    stats['sum_prompt']+=tokenized['prompt_tokens'];stats['sum_target']+=tokenized['target_tokens']
                    mask_checks+=1
                    if len(sample_batch)<4:sample_batch.append(tokenized)
            processed+=1
            if processed%100==0:print(f'Prepared {processed}/{len(memberships)} source samples',flush=True)
    if processed!=len(memberships):raise ValueError('selected metadata coverage mismatch')
    if any(totals[p]['rows']==0 for p in totals):raise ValueError('empty retained partition; inspect quarantine')
    collate_actions(sample_batch,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    verify_sources(root,manifest)
    report={'schema_version':1,'status':'complete','protocol':'custom_json','policy':'off',
            'training_performed':False,'toolenv_used':False,'outer_train_samples':len(manifest['train_ids']),
            'outer_dev_samples_excluded':len(manifest['dev_ids']),
            'inner_reserved':{'train':len(split['train_ids']),'validation':len(split['validation_ids']),
                              'seen_ids':len(split['seen_ids']),'forced_train_ids':len(split['forced_train_ids'])},
            'partitions':{p:dict(c) for p,c in totals.items()},'targets_by_tool':{p:dict(c) for p,c in tools.items()},
            'token_lengths':lengths,'mask_checks_passed':mask_checks,'quarantine_reasons':dict(rejected),
            'budgets':{'context_tokens':16384,'generation_new_tokens':384,'training_max_length':max_length,
                       'max_calls':8,'max_decisions':12},
            'overflow_policy':overflow_policy,'elapsed_seconds':time.perf_counter()-started,
            'limitations':['Retrospective inner validation; train was previously inspected.',
              'Grouping inherits existing heuristics; no guarantee of semantic deduplication.',
              'Independent decision examples are unweighted; long trajectories contribute more rows.',
              'Teacher-forced prefixes, not closed-loop model quality or ASR input.']}
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True,help='Local tokenizer; weights are not loaded')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seen-ids',type=Path,action='append',default=[])
    p.add_argument('--inner-split',type=Path,help='fixed inner split JSON (train_ids, validation_ids, outer_dev_ids); replaces --seen-ids/--seed/--validation-fraction')
    p.add_argument('--validation-fraction',type=float,default=.1)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--max-length',type=int,default=16384)
    p.add_argument('--overflow-policy',choices=['error','quarantine'],default='quarantine')
    a=p.parse_args(argv)
    if a.inner_split and (a.seen_ids or a.seed!=43 or a.validation_fraction!=.1):p.error('--inner-split excludes --seen-ids, --seed and --validation-fraction')
    root,out,model=safe_path(a.data_root),safe_path(a.output_dir),safe_path(a.model_path)
    manifest_path=safe_path(a.manifest);manifest_bytes=manifest_path.read_bytes();manifest=_parse(manifest_bytes.decode('utf-8-sig'))
    seen=set();seen_hashes=[]
    for path in a.seen_ids:
        path=safe_path(path);ids=_parse(path.read_text(encoding='utf-8-sig'))
        if not isinstance(ids,list) or not all(isinstance(i,str) and i for i in ids) or len(ids)!=len(set(ids)):
            raise ValueError('seen-ids must contain a unique string ID list')
        seen.update(ids);seen_hashes.append(file_hash(path))
    supplied=None;supplied_hash=None
    if a.inner_split:
        inner_path=safe_path(a.inner_split);supplied=_parse(inner_path.read_text(encoding='utf-8-sig'));supplied_hash=file_hash(inner_path)
    verify_sources(root,manifest)
    if not model.is_dir():raise ValueError('local tokenizer directory missing')
    if out.exists() or out.is_relative_to(root):raise ValueError('choose a new output outside Data')
    from transformers import AutoTokenizer
    from training.qwen35 import patch_tokenizer_loading
    patch_tokenizer_loading()  # no-op unless the tokenizer config is Qwen3.5
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    provenance=collect_provenance(Path(__file__).resolve().parents[1],config={
        'workflow':'prepare-trajectories','protocol':'custom_json','policy':'off','seed':a.seed,
        'validation_fraction':a.validation_fraction,'max_length':a.max_length,
        'seen_ids_sha256':seen_hashes,'inner_split_input_sha256':supplied_hash,'overflow_policy':a.overflow_policy},
        manifest=manifest_path,model_path=model,prompt=SYSTEM)
    report=build_dataset(root,manifest,tokenizer,out,seen_ids=seen,validation_fraction=a.validation_fraction,
                         seed=a.seed,max_length=a.max_length,overflow_policy=a.overflow_policy,explicit_split=supplied)
    report['provenance']=provenance
    # Detect manifest edits, not only source-data edits, during generation.
    if manifest_path.read_bytes()!=manifest_bytes:raise ValueError('manifest changed during build')
    write_json(out/'report.json',report)
    files={path.name:file_hash(path) for path in out.iterdir() if path.is_file()}
    write_json(out/'dataset_manifest.json',{'schema_version':1,'status':'complete',
        'files_sha256':files,'protocol':'custom_json','policy':'off',
        'outer_manifest_sha256':sha256(manifest_bytes).hexdigest(),
        'inner_split_sha256':file_hash(out/'inner_split.json'),
        'prompt_sha256':sha256(SYSTEM.encode()).hexdigest()})
    print(json.dumps({k:v for k,v in report.items() if k not in {'provenance','targets_by_tool'}},indent=2))


if __name__=='__main__':main()
