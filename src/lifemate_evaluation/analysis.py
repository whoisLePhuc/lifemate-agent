"""Frozen dev diagnostics; optional separate gold-only replay to verify state hashes."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def signature(call):
    return json.dumps(call,sort_keys=True,ensure_ascii=False,separators=(',',':'))


def indexed(rows, label):
    result = {}
    for row in rows:
        key = row.get('id') if isinstance(row, dict) else None
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f'{label}: missing or duplicate ID')
        result[key] = row
    return result


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--smoke-report',type=Path,required=True)
    p.add_argument('--verify-gold-hashes',action='store_true')
    p.add_argument('--output-name',default='analysis.json')
    a=p.parse_args(argv)
    a.data_root=a.data_root.resolve()
    a.run_dir=a.run_dir.resolve()
    if a.run_dir.is_relative_to(a.data_root):raise ValueError('analysis output must be outside Data')
    if Path(a.output_name).name!=a.output_name or a.output_name in ('.','..'):
        raise ValueError('output-name must be a filename')
    output=(a.run_dir/a.output_name).resolve()
    if output.parent!=a.run_dir or output.exists():raise ValueError('choose a new output inside run-dir')
    report=json.loads((a.run_dir/'report.json').read_text(encoding='utf-8'))
    metrics=indexed(report['rows'],'report')
    predictions=indexed(map(json.loads,(a.run_dir/'submission.jsonl').read_text(encoding='utf-8').splitlines()),'submission')
    if report['samples']!=len(metrics) or set(predictions)!=set(metrics) or not metrics:
        raise ValueError('report/submission coverage mismatch')
    refs={}
    with (a.data_root/'train/train.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['id'] in predictions:
                if r['id'] in refs:raise ValueError('duplicate train ID')
                refs[r['id']]=r
    if set(refs)!=set(predictions):raise ValueError('prediction ID absent from train references')
    smoke=json.loads(a.smoke_report.read_text(encoding='utf-8'))
    smoke_rows=indexed(smoke['rows'],'smoke')
    if not set(metrics).issubset(smoke_rows):raise ValueError('smoke coverage mismatch')
    smoke_states={r['id'] for r in smoke['rows'] if r['full_state_match']}
    buckets=defaultdict(Counter);totals=Counter();rows=[];apps=defaultdict(Counter)
    for metric in report['rows']:
        id=metric['id'];ref=refs[id];gold=ref['label']['tool_calls'];pred=predictions[id]['tool_calls']
        gc,pc=Counter(map(signature,gold)),Counter(map(signature,pred))
        gn,pn=Counter(c['name'] for c in gold),Counter(c['name'] for c in pred)
        exact=gc==pc;name_match=sum((gn&pn).values());arg_match=sum((gc&pc).values())
        g_ask='sys.ask_missing_info' in gn;p_ask='sys.ask_missing_info' in pn
        row={'id':id,'stop':metric['stop'],'state_match':metric['full_state_match'],
             'gold_state_changes':bool(ref['label']['final_state']),
             'gold_calls':len(gold),'pred_calls':len(pred),'exact_unordered_calls':exact,
             'name_matches':name_match,'exact_argument_matches':arg_match,
             'missing_tool_names':list((gn-pn).elements()),'extra_tool_names':list((pn-gn).elements()),
             'repeated_call_excess':sum(max(0,count-max(gc[k],1)) for k,count in pc.items()),
             'gold_asks':g_ask,'pred_asks':p_ask,'history':bool(ref['history']),
             'gold_names':[c['name'] for c in gold],'pred_names':[c['name'] for c in pred]}
        if a.verify_gold_hashes:
            # Independent evaluation run, one official ToolEnv per sample.
            # Never reset/retry an agent session or replace its recorded hash.
            from lifemate_tools import ToolEnvAdapter
            with ToolEnvAdapter(a.data_root,'train',id) as env:
                for call in gold:env.call(call['name'],call['arguments'])
                expected_hash=env.finalize()['state_hash']
            row['gold_hash_match']=predictions[id]['state_hash']==expected_hash
            totals['gold_hash_matches']+=row['gold_hash_match']
            totals['hash_and_exact_unordered_calls']+=row['gold_hash_match'] and exact
        rows.append(row)
        totals.update({'gold_calls':len(gold),'pred_calls':len(pred),'name_matches':name_match,
            'exact_argument_matches':arg_match,'exact_unordered_call_samples':exact,
            'state_and_exact_unordered_calls':exact and metric['full_state_match'],
            'name_multiset_matches':gn==pn,'zero_pred_calls':not pred,
            'empty_gold_correct':not gold and not pred,'empty_gold_total':not gold,
            'state_better_than_smoke':metric['full_state_match'] and id not in smoke_states,
            'state_worse_than_smoke':not metric['full_state_match'] and id in smoke_states,
            'state_same_correct_as_smoke':metric['full_state_match'] and id in smoke_states,
            'ask_tp':g_ask and p_ask,'ask_fn':g_ask and not p_ask,'ask_fp':p_ask and not g_ask,
            'samples_with_missing_tool_names':bool(gn-pn),'samples_with_extra_tool_names':bool(pn-gn),
            'samples_with_repeat_excess':row['repeated_call_excess']>0,
            'samples_with_validation_or_parse_errors':metric['errors']>0})
        for key in [f"stop:{metric['stop']}",f"history:{bool(ref['history'])}",f"gold_calls:{len(gold)}",
                    f"state_changes:{bool(ref['label']['final_state'])}"]:
            buckets[key].update({'samples':1,'state_match':metric['full_state_match'],
                                'exact_unordered':exact,'errors':metric['errors']})
            if a.verify_gold_hashes:buckets[key]['gold_hash_match']+=row['gold_hash_match']
        for app in {c['name'].split('.')[0] for c in gold}:
            apps[app].update({'samples':1,'state_match':metric['full_state_match'],'exact_unordered':exact})
    lat=sorted(r['elapsed_seconds'] for r in report['rows'])
    result={'samples':len(rows),'totals':dict(totals),'buckets':dict(buckets),'by_gold_app':dict(sorted(apps.items())),
            'latency_seconds':{'p50':lat[len(lat)//2],'p95':lat[int((len(lat)-1)*.95)],'max':max(lat),'sum':sum(lat)},
            'method':'Exact unordered argument equality; no fuzzy match or official competition score. Counts may overlap.',
            'rows':rows}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','by_gold_app']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
