"""Independent verification of a completed run before any number is reported.

Re-audits the trace against the submission, checks that independent replay hashes equal
the submission's hashes, recomputes every local score from replayed tool calls and
compares it with the stored score, then summarises the run. Integrated from
the former Experiments/019-history-context/run_validation.py (tag archive/experiments-011-021 in the workspace repository).
"""
import json
from pathlib import Path

from evaluation.scoring import score_sample
from lifemate.observability.trace import audit_trace


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def index(rows, ids):
    result = {r['id']: r for r in rows}
    if len(rows) != len(ids) or set(result) != set(ids):
        raise ValueError('missing/duplicate/unexpected sample IDs')
    return result


def ask_signature(calls):
    return {(c['arguments'].get('tool_name'), tuple(sorted(c['arguments'].get('params', []))))
            for c in calls if c['name'] == 'sys.ask_missing_info'}


def verify_run(run, ids, examples):
    """Return (summary, predictions, stored scores) or raise if any check fails."""
    run = Path(run)
    inf, sc = run / 'inference', run / 'scores'
    pred = index(lines(inf / 'submission.jsonl'), ids)
    stored = index(read(sc / 'scores.json')['rows'], ids)
    metrics = index(read(inf / 'report.json')['rows'], ids)
    replayed = index(read(sc / 'replay-predictions.json')['rows'], ids)
    gold_replay = index(read(sc / 'replay-gold.json')['rows'], ids)
    if not audit_trace(inf / 'trace.jsonl', list(pred.values()))['valid']:
        raise ValueError('fresh trace audit failed')
    for sid in ids:
        e, x = examples[sid], replayed[sid]
        if x['replayed_hash'] != pred[sid]['state_hash']:
            raise ValueError('replay hash differs from submission')
        score = score_sample(pred[sid]['tool_calls'], e.reference.tool_calls, e.agent_input.tools,
                             predicted_hash=x['replayed_hash'], gold_hash=gold_replay[sid]['replayed_hash'],
                             rejected_indices=x['schema_rejected_indices'])
        for key in ['local_score', 'hash_match', 'pairs', 'matched_calls']:
            if json.loads(json.dumps(score[key])) != stored[sid][key]:
                raise ValueError('fresh score mismatch')
    tokens = sum(e['data'].get('generation', {}).get('prompt_tokens', 0)
                 for e in lines(inf / 'trace.jsonl') if e['event'] == 'generation')
    lat = sorted(r['elapsed_seconds'] for r in metrics.values())
    ga = {s: ask_signature(examples[s].reference.tool_calls) for s in ids}
    pa = {s: ask_signature(pred[s]['tool_calls']) for s in ids}
    replay_report = read(sc / 'replay-predictions.json')
    summary = {
        'samples': len(ids),
        'mean_local_score': sum(x['local_score'] for x in stored.values()) / len(ids),
        'hash_matches': sum(x['hash_match'] for x in stored.values()),
        'prompt_tokens_total': tokens,
        'mean_seconds': sum(lat) / len(lat), 'p95_seconds': lat[int((len(lat) - 1) * .95)], 'max_seconds': max(lat),
        'tool_calls': sum(len(p['tool_calls']) for p in pred.values()),
        'schema_rejections': replay_report['schema_rejections'],
        'business_failures': replay_report['business_failures'],
        'stop_reasons': read(inf / 'report.json')['stop_reasons'],
        'ask_presence_tp': sum(bool(ga[s]) and bool(pa[s]) for s in ids),
        'ask_presence_fn': sum(bool(ga[s]) and not pa[s] for s in ids),
        'ask_presence_fp': sum(not ga[s] and bool(pa[s]) for s in ids),
        'ask_correct_tool_slots': sum(bool(ga[s] & pa[s]) for s in ids)}
    return summary, pred, stored
