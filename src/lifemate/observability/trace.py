"""Append-only private diagnostics. Never a substitute for the ToolEnv ledger."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode()).hexdigest()


class TraceWriter:
    def __init__(self, path, run_id, *, durable=True):
        self.path = Path(path)
        self.run_id = run_id
        self.durable = durable
        self.sequence = 0
        self.start = time.perf_counter()
        self.stream = self.path.open('x', encoding='utf-8', newline='\n')

    def emit(self, event, *, sample_id=None, **fields):
        row = {'schema_version': 1, 'run_id': self.run_id, 'sequence': self.sequence,
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'run_elapsed_seconds': time.perf_counter() - self.start,
               'sample_id': sample_id, 'event': event, 'data': fields}
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
        if self.durable:
            os.fsync(self.stream.fileno())
        self.sequence += 1

    def bind(self, sample_id):
        return lambda event, **fields: self.emit(event, sample_id=sample_id, **fields)

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def audit_trace(path, records):
    """Detect interrupted samples/calls and compare trace calls to real records.

    Does not infer success from a missing result. ToolEnv ledger remains authority.
    """
    expected = {r['id']: r for r in records}
    if len(expected) != len(records): raise ValueError('duplicate expected record ID')
    samples = {}
    issues = []
    run_id = None
    ended = False
    with Path(path).open(encoding='utf-8') as stream:
        for sequence, line in enumerate(stream):
            try:
                row = json.loads(line)
            except ValueError:
                issues.append('invalid_or_truncated_json'); break
            if not isinstance(row, dict):
                issues.append('invalid_json_record'); break
            if row.get('sequence') != sequence or row.get('schema_version') != 1:
                issues.append('invalid_sequence_or_schema')
            if sequence == 0:
                run_id = row.get('run_id')
                if row.get('event') != 'run_start': issues.append('missing_run_start')
            if row.get('run_id') != run_id or ended:
                issues.append('invalid_run_boundary')
            event, data, sid = row.get('event'), row.get('data', {}), row.get('sample_id')
            if not isinstance(data, dict) or (sid is not None and not isinstance(sid, str)):
                issues.append('invalid_event_fields'); break
            if event == 'run_end': ended = True
            if sid is None: continue
            sample = samples.setdefault(sid, {'started': False, 'ended': False, 'pending': None, 'calls': []})
            if sample['ended']: issues.append(f'{sid}:event_after_sample_end')
            if event == 'sample_start':
                if sample['started']: issues.append(f'{sid}:duplicate_sample_start')
                sample['started'] = True
            elif not sample['started']: issues.append(f'{sid}:missing_sample_start')
            if event == 'call_start':
                if sample['pending'] is not None: issues.append(f'{sid}:overlapping_calls')
                sample['pending'] = data.get('action')
            elif event == 'call_result':
                action = data.get('action')
                if sample['pending'] is None or action != sample['pending']:
                    issues.append(f'{sid}:unpaired_call_result')
                if isinstance(action, dict):
                    sample['calls'].append({'name': action.get('name'), 'arguments': action.get('arguments')})
                sample['pending'] = None
            elif event == 'sample_end':
                sample['ended'] = True
                record = expected.get(sid)
                if record is None or data.get('record_sha256') != fingerprint(record):
                    issues.append(f'{sid}:record_fingerprint_mismatch')
    if not ended: issues.append('missing_run_end')
    if set(samples) != set(expected): issues.append('sample_coverage_mismatch')
    for sid, sample in samples.items():
        if not sample['ended']: issues.append(f'{sid}:incomplete_sample')
        if sample['pending'] is not None: issues.append(f'{sid}:indeterminate_call')
        if sid in expected and sample['calls'] != expected[sid]['tool_calls']:
            issues.append(f'{sid}:ledger_mismatch')
    return {'valid': not issues, 'samples': len(samples), 'issues': issues}
