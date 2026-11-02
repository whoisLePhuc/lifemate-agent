"""Replay and locally score frozen train/dev predictions without model inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from evaluation.replay_predictions import read_predictions, selected_examples, dataset_fingerprints
from evaluation.scoring import score_sample, SCORER_VERSION, CONVENTIONS
from lifemate.observability.provenance import collect_provenance, file_hash


def evaluate(predictions, examples, predicted_replay, gold_replay):
    """Require complete, reproducible independent replay before producing a score."""
    def index(report, mode):
        rows = report['rows']
        indexed = {r['id']: r for r in rows}
        if report['mode'] != mode or len(indexed) != len(rows) or set(indexed) != {p['id'] for p in predictions}:
            raise ValueError('invalid replay coverage/mode')
        return indexed
    pred = index(predicted_replay, 'predictions')
    gold = index(gold_replay, 'gold')
    rows = []
    for record in predictions:
        sid = record['id']
        if not pred[sid]['hash_reproduced'] or not pred[sid]['ledger_matches'] or not gold[sid]['ledger_matches']:
            raise ValueError('prediction hash/ledger is not reproducible')
        if gold[sid]['schema_rejected_indices']:
            raise ValueError('gold replay has schema rejection; inspect toolkit/reference version')
        ex = examples[sid]
        score = score_sample(record['tool_calls'], ex.reference.tool_calls, ex.agent_input.tools,
                             predicted_hash=pred[sid]['replayed_hash'], gold_hash=gold[sid]['replayed_hash'],
                             rejected_indices=pred[sid]['schema_rejected_indices'])
        rows.append({'id': sid, **score})
    return {'scorer_version': SCORER_VERSION, 'official_score': False, 'conventions': CONVENTIONS,
            'samples': len(rows), 'mean_local_score': sum(r['local_score'] for r in rows)/len(rows),
            'hash_matches': sum(r['hash_match'] for r in rows),
            'matched_calls': sum(r['matched_calls'] for r in rows),
            'gold_calls': sum(r['gold_calls'] for r in rows),
            'predicted_calls': sum(r['predicted_calls'] for r in rows),
            'positive_score_samples': sum(r['local_score'] > 0 for r in rows), 'rows': rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--submission', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--partition',choices=['train','dev'],default='dev')
    p.add_argument('--allow-subset', action='store_true')
    a = p.parse_args(argv)
    root, out = a.data_root.resolve(), a.output_dir.resolve()
    if out.is_relative_to(root) or any(x.lower().endswith('_environment') for x in out.parts):
        p.error('outputs must be outside Data/environment')
    if out.exists(): raise FileExistsError('choose a new output directory')
    predictions = read_predictions(a.submission)
    examples = selected_examples(root, [r['id'] for r in predictions])
    if a.manifest:
        manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
        dev, train = manifest['dev_ids'], manifest['train_ids']
        ids = {r['id'] for r in predictions}
        if len(dev) != len(set(dev)) or set(dev) & set(train): raise ValueError('invalid manifest')
        selected = train if a.partition=='train' else dev
        if len(selected)!=len(set(selected)): raise ValueError('duplicate partition IDs')
        if not ids <= set(selected) or (not a.allow_subset and ids != set(selected)):
            raise ValueError('prediction coverage differs from frozen partition (use --allow-subset explicitly)')
        for name, digest in manifest['source_sha256'].items():
            source = (root/name).resolve()
            if not source.is_relative_to(root) or any(x.lower().endswith('_environment') for x in source.parts):
                raise ValueError('forbidden manifest source')
            if file_hash(source) != digest: raise ValueError('manifest source fingerprint mismatch')
    out.mkdir(parents=True, exist_ok=False)
    submission_hash = file_hash(a.submission)
    inputs = dataset_fingerprints(root)
    provenance = collect_provenance(Path(__file__).resolve().parents[1],
                                    config={'workflow':'evaluate-run','partition':a.partition, 'scorer_version':SCORER_VERSION},
                                    manifest=a.manifest)
    reports = {}
    for mode in ('predictions', 'gold'):
        output = out/f'replay-{mode}.json'
        # Process isolation is intentional: no second ToolEnv for an ID in one worker.
        subprocess.run([sys.executable, '-m', 'evaluation.replay_predictions',
                        '--data-root', str(root), '--submission', str(a.submission.resolve()),
                        '--output', str(output), '--mode', mode], check=True)
        reports[mode] = json.loads(output.read_text(encoding='utf-8'))
        if reports[mode]['submission_sha256'] != submission_hash or reports[mode]['inputs_sha256'] != inputs:
            raise RuntimeError('replay inputs changed between workers')
    if file_hash(a.submission) != submission_hash or dataset_fingerprints(root) != inputs:
        raise RuntimeError('evaluation inputs changed')
    result = evaluate(predictions, examples, reports['predictions'], reports['gold'])
    result.update({'submission_sha256': submission_hash, 'provenance': provenance,
                   'inputs_sha256': inputs})
    with (out/'scores.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in {'rows','inputs_sha256','provenance'}}, indent=2))


if __name__ == '__main__': main()
