"""Stage 2: transcripts -> Qwen3.5-4B + LoRA adapter (merged) -> ToolEnv -> submission.

Two input modes with identical model, controller and budgets:
- asr:    request and user-history text come only from a stage-1 transcripts file.
- oracle: train gold transcripts (evaluation control only; never allowed on test splits).

Train/dev runs reuse the lifemate text runner (evaluator, trace, replay contract) on the logical dev
partition. Test splits run the same controller in a local loop with one ToolEnv per sample and no
reference access at all. Needs the Qwen3.5 stack (transformers>=5, peft) and a GPU.
"""
import argparse, hashlib, importlib.metadata, json, time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from lifemate.observability.provenance import file_hash
from lifemate.agent.policy import READ_POLICIES
from lifemate.agent.protocol import LANGS

CONFIG = {'protocol': 'custom_json', 'policy': 'off', 'seconds': 30.0}
LIMITS = {'max_decisions': 12, 'max_calls': 8, 'max_repairs': 2}


def load_transcripts(path):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {'id', 'request', 'history'} or not isinstance(row['request'], str) \
                or not isinstance(row['history'], list) or not all(isinstance(h, str) for h in row['history']):
            raise ValueError('invalid transcript row')
        if row['id'] in rows:
            raise ValueError('duplicate transcript id')
        rows[row['id']] = row
    return rows


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def input_record(sample_id, payload):
    """Hashes of the text actually forwarded to the planner, for later audit."""
    users = [h['text'] for h in payload['history'] if h['role'] == 'user']
    return {'id': sample_id, 'request_sha256': text_sha(payload['request']),
            'history_sha256': [text_sha(t) for t in users]}


def journal_to(path):
    def write(row):
        with Path(path).open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
    return write


def asr_text_input(original, transcripts, journal=None):
    """Replace both transcript arguments; the caller's gold strings are discarded."""
    def build(item, _gold_request, _gold_history):
        row = transcripts[item.id]
        payload = original(item, row['request'], tuple(row['history']))
        if journal is not None:
            journal(input_record(item.id, payload))
        return payload
    return build


def load_planner(model_path, adapter_path, *, thinking=False, think_budget=1024):
    import torch
    from lifemate.agent.qwen35_planner import Qwen35Planner
    planner = Qwen35Planner(model_path, adapter_path=adapter_path, thinking=thinking, think_budget=think_budget)
    if adapter_path is not None:
        planner.model = planner.model.merge_and_unload().eval()
    planner.parameters = sum(p.numel() for p in planner.model.parameters())
    if planner.parameters >= 5_000_000_000:
        raise ValueError('merged model exceeds parameter limit')
    saved = planner.new_tokens; planner.new_tokens = 8
    saved_thinking, planner.thinking = planner.thinking, False   # warm-up must not enter thinking
    planner.generate([{'role': 'user', 'content': 'Reply with one short word.'}], 60)
    planner.new_tokens = saved; planner.thinking = saved_thinking
    torch.cuda.synchronize()
    return planner


def backend_info(planner, adapter_path):
    packages = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft']}
    if adapter_path is None:
        return {'model': 'Qwen3.5-4B base (no adapter; evaluation control)', 'adapter': None,
                'loaded_parameters': planner.parameters, 'packages': packages}
    adapter = Path(adapter_path)
    return {'model': 'Qwen3.5-4B + merged LoRA adapter', 'adapter': str(adapter),
            'adapter_files_sha256': {n: file_hash(adapter / n) for n in ('adapter_config.json', 'adapter_model.safetensors')},
            'loaded_parameters': planner.parameters,
            'packages': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft']}}


def run_dev(args, planner, out):
    import evaluation.run_text_baseline as runner
    import lifemate.agent.planner as pm
    pm.LocalPlanner = lambda *x, **k: planner
    journal = journal_to(out / 'inputs_used.jsonl')
    if args.mode == 'asr':
        runner.text_input = asr_text_input(runner.text_input, load_transcripts(args.transcripts), journal)
    else:
        gold_input = runner.text_input

        def oracle(item, request, history):
            payload = gold_input(item, request, history)
            journal(input_record(item.id, payload))
            return payload
        runner.text_input = oracle
    original = runner.collect_provenance

    def provenance(repo, *, config, **kwargs):
        return original(repo, config={**config, 'pipeline': 'asr-llm', 'input_mode': args.mode,
                                      'transcripts_sha256': file_hash(args.transcripts) if args.transcripts else None,
                                      'merged': bool(args.adapter_path), 'thinking': 'enabled' if planner.thinking else 'disabled'}, **kwargs)
    runner.collect_provenance = provenance
    runner.main(['--data-root', str(args.data_root), '--manifest', str(args.manifest), '--model-path', str(args.model_path),
                 *(['--adapter-path', str(args.adapter_path)] if args.adapter_path else []), '--partition', args.partition, '--ids-file', str(args.ids_file),
                 '--output-dir', str(out / 'inference'), '--protocol', args.protocol,
                 '--policy', args.policy, '--seconds', str(args.seconds), '--prompt-lang', args.prompt_lang,
                 '--thinking', 'on' if planner.thinking else 'off',
                 '--think-budget', str(planner.think_budget if planner.thinking else 0)])


def run_split(items, transcripts, planner, make_env, out, *, split, backend, full_split, seconds=CONFIG['seconds'],
              policy=CONFIG['policy']):
    """Production loop: one ToolEnv per sample, transcripts only, no reference data anywhere.

    `make_env(sample_id)` must return a ToolEnvAdapter-like context manager. Writes the submission,
    a durable trace and a report, and fails unless the trace audit is valid.
    """
    from lifemate.agent.controller import run_controller
    from lifemate.context.manager import text_input
    from lifemate.contracts import Outcome
    from lifemate.observability.trace import TraceWriter, audit_trace, fingerprint
    from lifemate.output.writer import SubmissionWriter, validate_submission
    ids = [it.id for it in items]
    if set(ids) != set(transcripts):
        raise ValueError('transcripts must cover exactly the selected samples')
    inf = out / 'inference'
    inf.mkdir(parents=True)
    journal = journal_to(out / 'inputs_used.jsonl')
    provenance = {'pipeline': 'asr-llm', 'split': split, 'input_mode': 'asr', **CONFIG, **LIMITS,
                  'seconds': seconds, 'policy': policy, **backend}
    (inf / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    rows, records = [], []
    with TraceWriter(inf / 'trace.jsonl', inf.name) as trace, SubmissionWriter(inf / 'submission.jsonl', ids) as writer:
        trace.emit('run_start', provenance=provenance)
        for item in items:
            started = time.perf_counter()
            emit = trace.bind(item.id); emit('sample_start')
            row = transcripts[item.id]
            payload = text_input(item, row['request'], tuple(row['history']))
            journal(input_record(item.id, payload))
            with make_env(item.id) as env:
                t0 = time.monotonic()
                try:
                    outcome = run_controller(payload, env, planner, seconds=seconds, trace=emit,
                                             protocol=CONFIG['protocol'], policy=policy)
                    failure = None
                except (ValueError, RuntimeError) as error:
                    outcome = getattr(error, 'controller_outcome', None) or \
                        Outcome('error', time.monotonic() - t0, 0, 0, len(env.get_tool_calls()))
                    failure = type(error).__name__
                    emit('controller_error', error_type=failure, outcome=asdict(outcome))
                record = env.finalize()
                emit('finalize', record_sha256=fingerprint(record))
            writer.write(record); records.append(record)
            rows.append({'id': item.id, **asdict(outcome), 'error_type': failure,
                         'total_sample_seconds': time.perf_counter() - started})
            with (inf / 'sample_results.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'metrics': rows[-1], 'record': record}, ensure_ascii=False) + '\n')
            emit('sample_end', stop=outcome.stop, record_sha256=fingerprint(record))
            print(f'{len(rows)}/{len(items)} {item.id}: {outcome.stop}', flush=True)
        writer.finish()
        trace.emit('run_end', samples=len(rows))
    audit = audit_trace(inf / 'trace.jsonl', records)
    (inf / 'trace_audit.json').write_text(json.dumps(audit, indent=2))
    if not audit['valid']:
        raise RuntimeError('trace/ledger audit failed')
    lat = sorted(r['elapsed_seconds'] for r in rows)
    report = {'split': split, 'samples': len(rows), 'full_split': full_split, 'trace_audit': audit,
              'submission_validation': validate_submission(inf / 'submission.jsonl', ids),
              'stop_reasons': dict(Counter(r['stop'] for r in rows)),
              'controller_errors': sum(r['error_type'] is not None for r in rows),
              'latency_seconds': {'mean': sum(lat) / len(lat), 'p95': lat[max(0, -(-95 * len(lat) // 100) - 1)], 'max': lat[-1]},
              'tool_calls': sum(len(r['tool_calls']) for r in records), 'score_computed': False}
    (inf / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'trace_audit'}, indent=2))
    return report


def run_test(args, planner, out):
    from lifemate.input.adapter import DatasetLoader
    from lifemate.tools.env_adapter import ToolEnvAdapter
    data_root = Path(args.data_root).resolve()
    transcripts = load_transcripts(args.transcripts)
    items = list(DatasetLoader(data_root, args.split).iter_inputs())
    if args.ids_file:  # bounded smoke on a subset; a submission needs the full split
        wanted = set(json.loads(args.ids_file.read_text()))
        items = [it for it in items if it.id in wanted]
        if {it.id for it in items} != wanted:
            raise ValueError('ids outside split')
    backend = {**backend_info(planner, args.adapter_path), 'transcripts_sha256': file_hash(args.transcripts)}
    return run_split(items, transcripts, planner, lambda sid: ToolEnvAdapter(data_root, args.split, sid), out,
                     split=args.split, backend=backend, full_split=not args.ids_file, seconds=args.seconds,
                     policy=args.policy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['train', 'public_test', 'private_test'], required=True,
                   help="'train' means the logical dev partition of the train data (needs --manifest)")
    p.add_argument('--mode', choices=['asr', 'oracle'], default='asr')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True, help='Qwen3.5-4B base model directory')
    p.add_argument('--adapter-path', type=Path, help='LoRA adapter directory (omit only for train/dev base-model controls)')
    p.add_argument('--transcripts', type=Path)
    p.add_argument('--ids-file', type=Path, help='dev IDs (train) or a smoke subset (test splits)')
    p.add_argument('--manifest', type=Path, help='split manifest that defines the dev partition (train only)')
    p.add_argument('--manifest-sha256', help='refuse to run if the manifest hash differs')
    p.add_argument('--seconds', type=float, default=CONFIG['seconds'], help='per-sample deadline')
    p.add_argument('--policy', choices=('off', *READ_POLICIES), default='off',
                   help='optional experimental repeated-read guard')
    p.add_argument('--protocol', choices=('custom_json', 'qwen35_native'), default=CONFIG['protocol'],
                   help='experimental: Qwen3.5 XML tool calling (evaluation on train/dev only)')
    p.add_argument('--prompt-lang', choices=LANGS, default='vi', help='experimental system-prompt language')
    p.add_argument('--thinking', choices=('off', 'on'), default='off', help='experimental Qwen3.5 thinking mode')
    p.add_argument('--think-budget', type=int, default=1024, help='thinking tokens before </think> is forced')
    p.add_argument('--partition', choices=('dev', 'train'), default='dev',
                   help="dev: logical outer dev. train: only the frozen 124 inner-validation IDs (--inner-split)")
    p.add_argument('--inner-split', type=Path, help='inner_split.json defining the 124 validation IDs (partition train)')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.exists():
        raise SystemExit('choose a new output directory')
    if a.mode == 'oracle' and (a.split != 'train' or a.transcripts):
        raise SystemExit('oracle mode is a train/dev evaluation control only')
    if a.mode == 'asr' and not a.transcripts:
        raise SystemExit('asr mode needs --transcripts')
    if a.split == 'train' and not (a.ids_file and a.manifest):
        raise SystemExit('train/dev run needs --ids-file and --manifest')
    if a.manifest and a.manifest_sha256 and file_hash(a.manifest) != a.manifest_sha256:
        raise SystemExit('split manifest hash changed: ' + str(a.manifest))
    if a.adapter_path is None and a.split != 'train':
        raise SystemExit('test splits require --adapter-path; base-model runs are train/dev controls only')
    experimental = (a.protocol, a.prompt_lang, a.thinking, a.partition) != (CONFIG['protocol'], 'vi', 'off', 'dev')
    if experimental and a.split != 'train':
        raise SystemExit('experimental protocol/language/thinking/partition options are train/dev evaluation only')
    if a.partition == 'train':
        if not (a.inner_split and a.ids_file):
            raise SystemExit('partition train needs --inner-split and --ids-file')
        validation = json.loads(a.inner_split.read_text()).get('validation_ids')
        if not isinstance(validation, list) or not set(json.loads(a.ids_file.read_text())) <= set(validation):
            raise SystemExit('partition train is restricted to inner-validation IDs')
    if a.thinking == 'on' and a.think_budget < 1:
        raise SystemExit('think-budget must be positive')
    a.data_root = a.data_root.resolve()
    # Load first: a failed model load must not leave an output directory that blocks the rerun.
    planner = load_planner(a.model_path, a.adapter_path, thinking=a.thinking == 'on', think_budget=a.think_budget)
    out.mkdir(parents=True)
    (out / 'backend.json').write_text(json.dumps(backend_info(planner, a.adapter_path), indent=2))
    (run_dev if a.split == 'train' else run_test)(a, planner, out)


if __name__ == '__main__':
    main()
