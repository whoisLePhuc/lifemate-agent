"""Train-only independent replay worker. Gold and predictions run in separate processes."""
import argparse
import json
from pathlib import Path
import time

from lifemate.input.adapter import DatasetLoader, _parse
from lifemate.output.writer import validate_submission
from lifemate.tools.env_adapter import ToolEnvAdapter
from lifemate.observability.provenance import file_hash


def read_predictions(path):
    if any(p.lower().endswith('_environment') for p in Path(path).resolve().parts):
        raise ValueError('environment paths are forbidden')
    rows = [_parse(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    ids = [r.get('id') for r in rows]
    if not rows: raise ValueError('empty submission')
    validate_submission(path, ids)
    return rows


def selected_examples(root, ids):
    wanted = set(ids)
    examples = {ex.agent_input.id: ex for ex in DatasetLoader(root, 'train', check_audio=False).iter_training()
                if ex.agent_input.id in wanted}
    if set(examples) != wanted: raise ValueError('submission contains non-train or unknown IDs')
    return examples


def dataset_fingerprints(root):
    root = Path(root)
    # Explicit metadata/toolkit paths only, never enumerate environment directories.
    files = [root/'train/train.jsonl', root/'train/train_tools_catalog.json']
    files += sorted((root/'train/zaic_tools').rglob('*.py'))
    for file in files:
        if not file.resolve().is_relative_to(root.resolve()) or any(
                p.lower().endswith('_environment') for p in file.resolve().parts):
            raise ValueError('unsafe toolkit/metadata path')
    return {p.relative_to(root).as_posix(): file_hash(p) for p in files}


def replay_records(root, predictions, *, mode='predictions', env_factory=None):
    """One session per ID. Never validate/filter away submitted rejected calls."""
    if mode not in {'predictions', 'gold'}: raise ValueError('invalid replay mode')
    ids = [r['id'] for r in predictions]
    if len(ids) != len(set(ids)): raise ValueError('duplicate replay ID')
    examples = selected_examples(root, ids)
    factory = env_factory or ToolEnvAdapter
    rows = []
    for prediction in predictions:
        sid = prediction['id']
        calls = prediction['tool_calls'] if mode == 'predictions' else examples[sid].reference.tool_calls
        started = time.perf_counter()
        rejected, failed = [], []
        with factory(root, 'train', sid) as env:
            for index, call in enumerate(calls):
                result = env.call(call['name'], call['arguments'])
                if result.get('ok') is False: rejected.append(index)
                if result.get('ok') is True and result.get('done') is False: failed.append(index)
            record = env.finalize()
        rows.append({'id': sid, 'replayed_hash': record['state_hash'],
                     'submitted_hash': prediction['state_hash'],
                     'hash_reproduced': record['state_hash'] == prediction['state_hash'],
                     'ledger_matches': record['tool_calls'] == calls, 'calls': len(calls),
                     'schema_rejected_indices': rejected, 'business_failure_indices': failed,
                     'elapsed_seconds': time.perf_counter()-started})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--submission', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['predictions', 'gold'], default='predictions')
    a = p.parse_args(argv)
    root, output = a.data_root.resolve(), a.output.resolve()
    if output.is_relative_to(root) or any(x.lower().endswith('_environment') for x in output.parts):
        p.error('replay output must be outside Data/environment')
    if output.exists(): raise FileExistsError('replay output exists')
    predictions = read_predictions(a.submission)
    inputs = dataset_fingerprints(root)
    rows = replay_records(root, predictions, mode=a.mode)
    if inputs != dataset_fingerprints(root): raise RuntimeError('dataset/toolkit changed during replay')
    result = {'schema_version': 1, 'mode': a.mode, 'split': 'train',
              'submission_sha256': file_hash(a.submission), 'inputs_sha256': inputs,
              'samples': len(rows), 'hashes_reproduced': sum(r['hash_reproduced'] for r in rows),
              'ledger_matches': sum(r['ledger_matches'] for r in rows),
              'schema_rejections': sum(len(r['schema_rejected_indices']) for r in rows),
              'business_failures': sum(len(r['business_failure_indices']) for r in rows), 'rows': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in {'rows','inputs_sha256'}}, indent=2))
    return int(result['ledger_matches'] != len(rows) or
               (a.mode == 'predictions' and result['hashes_reproduced'] != len(rows)))


if __name__ == '__main__': raise SystemExit(main())
