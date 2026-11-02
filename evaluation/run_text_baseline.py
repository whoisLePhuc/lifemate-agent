from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

"""Evaluate real local LLM or explicitly named finish-only plumbing smoke on frozen dev."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

from lifemate.input.adapter import DatasetLoader
from evaluation.replay import TrainReplaySession
from lifemate.output.writer import SubmissionWriter, validate_submission
from lifemate.agent.controller import run_controller
from lifemate.context.manager import text_input
from lifemate.agent.protocol import PROTOCOLS, system_prompt
from lifemate.agent.policy import POLICIES
from lifemate.observability.provenance import collect_provenance
from lifemate.observability.trace import TraceWriter, audit_trace, fingerprint


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model-path',type=Path)
    p.add_argument('--smoke-finish-only',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--protocol',choices=PROTOCOLS,default='custom_json')
    p.add_argument('--policy',choices=POLICIES,default='off')
    p.add_argument('--partition',choices=['train','dev'],default='dev')
    p.add_argument('--ids-file',type=Path,help='JSON list of IDs restricted to the selected partition')
    args=p.parse_args(argv)
    if bool(args.model_path)==args.smoke_finish_only: p.error('choose model-path OR smoke-finish-only')
    if args.limit is not None and args.limit<1: p.error('limit must be positive')
    if args.limit is not None and args.ids_file: p.error('use limit OR ids-file')
    if args.seconds<=0: p.error('seconds must be positive')
    root=args.data_root.resolve();out=args.output_dir.resolve()
    if out.is_relative_to(root): p.error('outputs must be outside Data')
    if out.exists(): p.error('choose a new output directory')
    manifest_bytes=args.manifest.read_bytes();manifest=json.loads(manifest_bytes)
    for name,digest in manifest['source_sha256'].items():
        source=(root/name).resolve()
        if not source.is_relative_to(root) or any(x.endswith('_environment') for x in source.parts): p.error('forbidden manifest source')
        if sha256(source.read_bytes()).hexdigest()!=digest: p.error('source fingerprint mismatch')
    train,dev=manifest['train_ids'],manifest['dev_ids']
    if len(dev)!=len(set(dev)) or set(train)&set(dev):p.error('invalid split IDs')
    partition_ids=train if args.partition=='train' else dev
    if len(partition_ids)!=len(set(partition_ids)):p.error('duplicate partition IDs')
    chosen=sorted(partition_ids)[:args.limit] if args.limit else sorted(partition_ids)
    ids_bytes=None
    if args.ids_file:
        if any(x.lower().endswith('_environment') for x in args.ids_file.resolve().parts):p.error('forbidden ids path')
        ids_bytes=args.ids_file.read_bytes();selected=json.loads(ids_bytes)
        if not isinstance(selected,list) or not all(isinstance(i,str) for i in selected):p.error('ids-file must be a JSON list of strings')
        if len(selected)!=len(set(selected)) or not set(selected)<=set(partition_ids):p.error('IDs must be unique and inside the selected partition')
        chosen=sorted(selected)
    if not chosen:p.error('empty selected set')
    prompt=system_prompt(args.protocol,args.policy)
    provenance=collect_provenance(Path(__file__).resolve().parents[1],
        config={'workflow':'text-baseline','protocol':args.protocol,'policy':args.policy,'partition':args.partition,'seed':42,'seconds':args.seconds,
                'max_decisions':12,'max_calls':8,'max_repairs':2,'context_tokens':16384,'new_tokens':384,
                'decoding':'greedy','dtype':'bfloat16','attention':'sdpa',
                'smoke_finish_only':args.smoke_finish_only,'chosen_ids_sha256':fingerprint(chosen),
                'ids_file_sha256':sha256(ids_bytes).hexdigest() if ids_bytes else None,
                'trace_durable':True}, manifest=args.manifest, model_path=args.model_path, prompt=prompt)
    if args.smoke_finish_only:
        class FinishOnly:
            def generate(self,messages,seconds,**kwargs):
                return 'Hoàn tất.' if args.protocol=='qwen_native' else '{"type":"finish"}'
        planner=FinishOnly()
    else:
        from lifemate.agent.planner import LocalPlanner
        planner=LocalPlanner(args.model_path)
        import torch
        provenance['hardware']={'gpu':torch.cuda.get_device_name(0),
                                'cuda_runtime':torch.version.cuda,'parameters':planner.parameters}
    out.mkdir(parents=True,exist_ok=False)
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[];records=[];wanted=set(chosen)
    with TraceWriter(out/'trace.jsonl',out.name) as trace, SubmissionWriter(out/'submission.jsonl',chosen) as writer:
        trace.emit('run_start',provenance=provenance)
        for ex in DatasetLoader(root,'train').iter_training():
            item,ref=ex.agent_input,ex.reference
            if item.id not in wanted:continue
            sample_start=time.perf_counter()
            emit=trace.bind(item.id)
            emit('sample_start')
            # Only transcript strings cross into planner input, never the reference object.
            payload=text_input(item,ref.audio_text,ref.history_audio_text)
            with TrainReplaySession(root,item.id) as env:
                controller_start=time.monotonic()
                try:
                    outcome=run_controller(payload,env,planner,seconds=args.seconds,trace=emit,protocol=args.protocol,policy=args.policy)
                    failure=None
                except (ValueError,RuntimeError) as error:
                    # Hashing after an indeterminate runtime failure will raise: no fake record.
                    from lifemate.contracts import Outcome
                    outcome=getattr(error,'controller_outcome',None)
                    if outcome is None:
                        outcome=Outcome('error',time.monotonic()-controller_start,0,0,len(env.get_tool_calls()))
                    failure=type(error).__name__
                    emit('controller_error',error_type=failure,outcome=asdict(outcome))
                evaluation_start=time.perf_counter()
                state=env.compare_final_state(ref.final_state)
                evaluation_seconds=time.perf_counter()-evaluation_start
                # Only evaluator timing is logged; no reference or app state enters trace.
                finalize_start=time.perf_counter()
                record=env.finalize()
                finalize_seconds=time.perf_counter()-finalize_start
                emit('finalize',elapsed_seconds=finalize_seconds,
                     timings=getattr(env,'last_finalize_metrics',{}),record_sha256=fingerprint(record))
            write_start=time.perf_counter()
            writer.write(record)
            write_seconds=time.perf_counter()-write_start
            records.append(record)
            rows.append({'id':item.id,**asdict(outcome),'error_type':failure,
                         'total_sample_seconds':time.perf_counter()-sample_start,
                         'evaluation_seconds':evaluation_seconds,'finalize_seconds':finalize_seconds,
                         'write_seconds':write_seconds,
                         'full_state_match':state['full_state_match'],
                         'exact_ordered_calls':record['tool_calls']==ref.tool_calls})
            # Close each append so live inspection never depends on buffered
            # submission output. Internal diagnostics only; not a submission.
            with (out/'sample_results.jsonl').open('a',encoding='utf-8') as journal:
                journal.write(json.dumps({'metrics':rows[-1],'record':record},ensure_ascii=False)+'\n')
            emit('sample_end',stop=outcome.stop,record_sha256=fingerprint(record),
                 total_sample_seconds=time.perf_counter()-sample_start,
                 evaluation_seconds=evaluation_seconds,write_seconds=write_seconds)
            print(f'{len(rows)}/{len(chosen)} {item.id}: {outcome.stop}',flush=True)
        publish_start=time.perf_counter()
        writer.finish()
        publish_seconds=time.perf_counter()-publish_start
        trace.emit('run_end',samples=len(rows),submission_publish_seconds=publish_seconds)
    trace_audit=audit_trace(out/'trace.jsonl',records)
    (out/'trace_audit.json').write_text(json.dumps(trace_audit,indent=2),encoding='utf-8')
    if not trace_audit['valid']: raise RuntimeError('trace/ledger audit failed; inspect trace_audit.json')
    summary=validate_submission(out/'submission.jsonl',chosen)
    terminal={'finished','asked_missing_info'}
    report={'mode':'finish_only_plumbing_smoke' if args.smoke_finish_only else 'local_llm_gold_transcript_'+args.partition,
            'protocol':args.protocol,'policy':args.policy,'partition':args.partition,
            'python':platform.python_version(),'command':sys.argv,'samples':len(rows),
            'provenance':provenance,'trace_audit':trace_audit,'submission_publish_seconds':publish_seconds,
            'model_path':str(args.model_path) if args.model_path else None,
            'model_parameters':getattr(planner,'parameters',None),
            'manifest_sha256':sha256(manifest_bytes).hexdigest(),'prompt_sha256':sha256(prompt.encode()).hexdigest(),
            'full_state_matches':sum(r['full_state_match'] for r in rows),
            'completed_state_matches':sum(r['full_state_match'] and r['stop'] in terminal for r in rows),
            'exact_ordered_call_matches':sum(r['exact_ordered_calls'] for r in rows),
            'stop_reasons':dict(Counter(r['stop'] for r in rows)), 'submission_validation':summary,
            'latency_seconds':{'mean':sum(r['elapsed_seconds'] for r in rows)/len(rows),
                               'max':max(r['elapsed_seconds'] for r in rows)},
            'competition_score_computed':False, 'rows':rows}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'rows','provenance'}},indent=2))


if __name__=='__main__':main()
