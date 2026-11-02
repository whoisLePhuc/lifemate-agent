"""Paired custom/native protocol diagnostic, train partition only; no test labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys

from lifemate.input.adapter import DatasetLoader
from evaluation.replay_predictions import read_predictions


def select_diagnostic_ids(examples, train_ids, dev_ids, *, samples=16, seed=42):
    train, dev = set(train_ids), set(dev_ids)
    if train & dev or len(train) != len(train_ids) or len(dev) != len(dev_ids):
        raise ValueError('invalid frozen partition')
    if samples < 4: raise ValueError('at least four diagnostic samples required')
    buckets = {name:[] for name in ('ask','out_of_scope','single','multi')}
    for ex in examples:
        sid = ex.agent_input.id
        if sid not in train: continue
        calls = ex.reference.tool_calls
        category = ('ask' if any(c['name']=='sys.ask_missing_info' for c in calls) else
                    'out_of_scope' if not calls else 'single' if len(calls)==1 else 'multi')
        buckets[category].append(sid)
    rng = random.Random(seed)
    for ids in buckets.values():
        ids.sort();rng.shuffle(ids)
    selected, distribution = [], Counter()
    while len(selected) < samples:
        progressed = False
        for category, ids in buckets.items():
            if ids and len(selected)<samples:
                selected.append(ids.pop());distribution[category]+=1;progressed=True
        if not progressed: raise ValueError('not enough training examples')
    if len(set(selected)) != len(selected): raise ValueError('duplicate diagnostic ID')
    return sorted(selected), dict(distribution)


def compare_reports(custom, native, custom_scores, native_scores, *, comparison='protocol'):
    """Compare only aligned IDs/configs. Scores remain local and uncertified."""
    def indexed(rows):
        values={r['id']:r for r in rows}
        if len(values)!=len(rows): raise ValueError('duplicate comparison ID')
        return values
    cm,nm=indexed(custom['rows']),indexed(native['rows'])
    cs,ns=indexed(custom_scores['rows']),indexed(native_scores['rows'])
    if not cm or not (set(cm)==set(nm)==set(cs)==set(ns)): raise ValueError('paired coverage mismatch')
    if custom_scores['scorer_version']!=native_scores['scorer_version'] or custom_scores['conventions']!=native_scores['conventions']:
        raise ValueError('scorer conventions differ')
    a,b=custom['provenance'],native['provenance']
    if a['model'] != b['model'] or a['manifest_sha256'] != b['manifest_sha256'] or a['packages'] != b['packages']:
        raise ValueError('model/data/packages differ')
    def runtime_sources(provenance):
        hashes=provenance.get('source',{}).get('source_files_sha256',{})
        return {k:v for k,v in hashes.items() if k.startswith(('src/lifemate/','evaluation/','training/'))
                or k=='pyproject.toml' or k.startswith('requirements')}
    if runtime_sources(a)!=runtime_sources(b): raise ValueError('runtime source changed between runs')
    keys=('seed','seconds','max_decisions','max_calls','max_repairs','context_tokens','new_tokens',
          'decoding','dtype','attention','chosen_ids_sha256','partition')
    if any(a['config'][k]!=b['config'][k] for k in keys): raise ValueError('paired generation budgets/IDs differ')
    if comparison=='protocol':
        if a['config']['protocol']!='custom_json' or b['config']['protocol']!='qwen_native':
            raise ValueError('wrong comparison protocols')
        if a['config'].get('policy','off')!='off' or b['config'].get('policy','off')!='off':
            raise ValueError('protocol comparison requires policy off')
        labels=('custom_json','qwen_native')
    elif comparison=='policy':
        if a['config']['protocol']!='custom_json' or b['config']['protocol']!='custom_json' or (
            a['config'].get('policy')!='off' or b['config'].get('policy')!='grounded_v1'):
            raise ValueError('policy comparison requires off vs grounded_v1 on custom_json')
        labels=('off','grounded_v1')
    else:raise ValueError('unknown comparison type')
    rows=[]
    for sid in sorted(cm):
        rows.append({'id':sid,'custom_local_score':cs[sid]['local_score'],'native_local_score':ns[sid]['local_score'],
                     'delta':ns[sid]['local_score']-cs[sid]['local_score'],
                     'custom_stop':cm[sid]['stop'],'native_stop':nm[sid]['stop']})
    def summary(report,scores):
        lat=sorted(r['elapsed_seconds'] for r in report['rows'])
        return {'mean_local_score':scores['mean_local_score'],'hash_matches':scores['hash_matches'],
                'predicted_calls':scores['predicted_calls'],'matched_calls':scores['matched_calls'],
                'stop_reasons':report['stop_reasons'],
                'controller_errors_total':sum(r['errors'] for r in report['rows']),
                'latency_seconds':{'mean':sum(lat)/len(lat),'p50':lat[len(lat)//2],
                                   'p95':lat[int((len(lat)-1)*.95)],'max':max(lat)}}
    if comparison=='policy':
        rows=[{'id':r['id'],'baseline_local_score':r['custom_local_score'],'candidate_local_score':r['native_local_score'],
               'delta':r['delta'],'baseline_stop':r['custom_stop'],'candidate_stop':r['native_stop']} for r in rows]
    return {'comparison':comparison,'samples':len(rows),'partition':a['config']['partition'],'official_score':False,
            labels[0]:summary(custom,custom_scores),labels[1]:summary(native,native_scores),
            'improved_samples':sum(r['delta']>0 for r in rows),'regressed_samples':sum(r['delta']<0 for r in rows),
            'unchanged_samples':sum(r['delta']==0 for r in rows),
            'limitations':['Training diagnostic, not held-out dev or full-dev quality estimate.',
                           'Protocol formatting changes together.' if comparison=='protocol' else 'Policy prompt, evidence guards and progress feedback change together.',
                           'Same model path/metadata, not a fresh checksum of all weight shards.',
                           'Custom runs first; timing is diagnostic, not a controlled speed benchmark.'],
            'rows':rows}


def main(argv=None,*,comparison='protocol'):
    p=argparse.ArgumentParser(description=__doc__ if comparison=='protocol' else 'Paired off/grounded policy diagnostic, train partition only')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--samples',type=int,default=16)
    a=p.parse_args(argv)
    root,out=a.data_root.resolve(),a.output_dir.resolve()
    if out.is_relative_to(root) or any(x.lower().endswith('_environment') for x in out.parts):
        p.error('outputs must be outside Data/environment')
    if out.exists(): raise FileExistsError('choose a new comparison directory')
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    ids,distribution=select_diagnostic_ids(DatasetLoader(root,'train',check_audio=False).iter_training(),
        manifest['train_ids'],manifest['dev_ids'],samples=a.samples)
    out.mkdir(parents=True,exist_ok=False)
    selection={'seed':42,'samples':len(ids),'partition':'train','distribution':distribution,
               'note':'Gold is used only to stratify diagnostic selection, never passed to the planner.'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    ids_path=out/'ids.json';ids_path.write_text(json.dumps(ids,indent=2),encoding='utf-8')
    reports,scores={},{}
    variants=('custom_json','qwen_native') if comparison=='protocol' else ('off','grounded_v1')
    for variant in variants:
        protocol=variant if comparison=='protocol' else 'custom_json'
        policy='off' if comparison=='protocol' else variant
        print(f'Running {variant}: {len(ids)} training diagnostic samples',flush=True)
        with (out/f'{variant}.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','-m','lifemate.cli','text-baseline',
                '--data-root',str(root),'--manifest',str(a.manifest.resolve()),'--model-path',str(a.model_path.resolve()),
                '--protocol',protocol,'--policy',policy,'--partition','train','--ids-file',str(ids_path),
                '--output-dir',str(out/variant)],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/f'{variant}-evaluation.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','lifemate.cli','evaluate-run','--data-root',str(root),
                '--submission',str(out/variant/'submission.jsonl'),'--manifest',str(a.manifest.resolve()),
                '--partition','train','--allow-subset','--output-dir',str(out/(variant+'-scores'))],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        reports[variant]=json.loads((out/variant/'report.json').read_text(encoding='utf-8'))
        scores[variant]=json.loads((out/(variant+'-scores')/'scores.json').read_text(encoding='utf-8'))
        if {r['id'] for r in reports[variant]['rows']}!=set(ids): raise ValueError('comparison selection differs')
    result=compare_reports(reports[variants[0]],reports[variants[1]],scores[variants[0]],scores[variants[1]],comparison=comparison)
    result['selection']=selection
    with (out/'comparison.json').open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
