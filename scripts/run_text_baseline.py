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

from lifemate_data import DatasetLoader
from lifemate_evaluation.replay import TrainReplaySession
from lifemate_output import SubmissionWriter, validate_submission
from lifemate_agent.text import run_controller,text_input,SYSTEM


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model-path',type=Path)
    p.add_argument('--smoke-finish-only',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--seconds',type=float,default=30)
    args=p.parse_args()
    if bool(args.model_path)==args.smoke_finish_only: p.error('choose model-path OR smoke-finish-only')
    if args.limit is not None and args.limit<1: p.error('limit must be positive')
    if args.seconds<=0: p.error('seconds must be positive')
    root=args.data_root.resolve();out=args.output_dir.resolve()
    if out.is_relative_to(root): p.error('outputs must be outside Data')
    manifest_bytes=args.manifest.read_bytes();manifest=json.loads(manifest_bytes)
    for name,digest in manifest['source_sha256'].items():
        source=(root/name).resolve()
        if not source.is_relative_to(root) or any(x.endswith('_environment') for x in source.parts): p.error('forbidden manifest source')
        if sha256(source.read_bytes()).hexdigest()!=digest: p.error('source fingerprint mismatch')
    train,dev=manifest['train_ids'],manifest['dev_ids']
    if len(dev)!=len(set(dev)) or set(train)&set(dev):p.error('invalid split IDs')
    chosen=sorted(dev)[:args.limit] if args.limit else sorted(dev)
    if not chosen:p.error('empty dev set')
    if args.smoke_finish_only:
        class FinishOnly:
            def generate(self,messages,seconds):return '{"type":"finish"}'
        planner=FinishOnly()
    else:
        from lifemate_agent.local_model import LocalPlanner
        planner=LocalPlanner(args.model_path)
    out.mkdir(parents=True,exist_ok=False)
    rows=[];wanted=set(chosen)
    with SubmissionWriter(out/'submission.jsonl',chosen) as writer:
        for ex in DatasetLoader(root,'train').iter_training():
            item,ref=ex.agent_input,ex.reference
            if item.id not in wanted:continue
            # Only transcript strings cross into planner input, never the reference object.
            payload=text_input(item,ref.audio_text,ref.history_audio_text)
            with TrainReplaySession(root,item.id) as env:
                sample_start=time.monotonic()
                try:
                    outcome=run_controller(payload,env,planner,seconds=args.seconds)
                    failure=None
                except (ValueError,RuntimeError) as error:
                    # Hashing after an indeterminate runtime failure will raise: no fake record.
                    from lifemate_agent.text import Outcome
                    outcome=Outcome('error',time.monotonic()-sample_start,0,0,len(env.get_tool_calls()));failure=type(error).__name__
                state=env.compare_final_state(ref.final_state)
                record=env.finalize()
            writer.write(record)
            rows.append({'id':item.id,**asdict(outcome),'error_type':failure,
                         'full_state_match':state['full_state_match'],
                         'exact_ordered_calls':record['tool_calls']==ref.tool_calls})
            # Close each append so live inspection never depends on buffered
            # submission output. Internal diagnostics only; not a submission.
            with (out/'sample_results.jsonl').open('a',encoding='utf-8') as journal:
                journal.write(json.dumps({'metrics':rows[-1],'record':record},ensure_ascii=False)+'\n')
            print(f'{len(rows)}/{len(chosen)} {item.id}: {outcome.stop}',flush=True)
    summary=validate_submission(out/'submission.jsonl',chosen)
    terminal={'finished','asked_missing_info'}
    report={'mode':'finish_only_plumbing_smoke' if args.smoke_finish_only else 'local_llm_gold_transcript_dev',
            'python':platform.python_version(),'command':sys.argv,'samples':len(rows),
            'model_path':str(args.model_path) if args.model_path else None,
            'model_parameters':getattr(planner,'parameters',None),
            'manifest_sha256':sha256(manifest_bytes).hexdigest(),'prompt_sha256':sha256(SYSTEM.encode()).hexdigest(),
            'full_state_matches':sum(r['full_state_match'] for r in rows),
            'completed_state_matches':sum(r['full_state_match'] and r['stop'] in terminal for r in rows),
            'exact_ordered_call_matches':sum(r['exact_ordered_calls'] for r in rows),
            'stop_reasons':dict(Counter(r['stop'] for r in rows)), 'submission_validation':summary,
            'latency_seconds':{'mean':sum(r['elapsed_seconds'] for r in rows)/len(rows),
                               'max':max(r['elapsed_seconds'] for r in rows)},
            'competition_score_computed':False, 'rows':rows}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
