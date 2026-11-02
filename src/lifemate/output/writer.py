"""Strict JSONL submission writing with coverage checks and atomic publication."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable


class SubmissionError(ValueError):
    pass


def _ids(values: Iterable[str]) -> set[str]:
    result = set()
    for value in values:
        if not isinstance(value, str) or not value.strip() or value in result:
            raise SubmissionError("expected IDs must be unique nonempty strings")
        result.add(value)
    return result


def _json_value(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_value(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_value(v) for v in value)
    return value is None or type(value) in (str, int, float, bool)


def _record(record: dict, expected: set[str], seen: set[str]) -> str:
    if not isinstance(record, dict) or set(record) != {"id", "tool_calls", "state_hash"}:
        raise SubmissionError("record must contain exactly id, tool_calls, state_hash")
    sample_id = record["id"]
    if not isinstance(sample_id, str) or sample_id not in expected:
        raise SubmissionError("unexpected sample ID")
    if sample_id in seen:
        raise SubmissionError("duplicate sample ID")
    if not isinstance(record["state_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["state_hash"]):
        raise SubmissionError("state_hash must be a lowercase SHA-256 hex string")
    calls = record["tool_calls"]
    if not isinstance(calls, list):
        raise SubmissionError("tool_calls must be an array")
    for call in calls:
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or not call["name"]
                or not isinstance(call["arguments"], dict)):
            raise SubmissionError("invalid tool-call structure")
    try:
        if not _json_value(record):
            raise SubmissionError("record contains non-JSON values")
        return json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise SubmissionError("record is not finite UTF-8 JSON") from exc


class SubmissionWriter:
    """Write records unchanged; publish only after all expected IDs are present.

    A sibling temporary file is flushed/fsynced, then linked atomically to the
    final path without replacing an existing file. Requires a filesystem with
    hard-link support (e.g. NTFS/ext4); fails explicitly otherwise. No reset,
    inference, tool filtering or hash computation happens here.
    """
    def __init__(self, output: str | Path, expected_ids: Iterable[str]):
        self.output = Path(output).resolve()
        self.expected = _ids(expected_ids)
        self.seen = set()
        self.status = "open"
        if self.output.exists():
            raise FileExistsError("submission already exists")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.output.name}.", suffix=".tmp", dir=self.output.parent)
        self._temp = Path(name)
        self._stream = os.fdopen(fd, "w", encoding="utf-8", newline="\n")

    def write(self, record: dict):
        if self.status != "open":
            raise SubmissionError("writer is not open")
        text = _record(record, self.expected, self.seen)
        try:
            self._stream.write(text + "\n")
        except (OSError, UnicodeError):
            self.abort()
            raise
        self.seen.add(record["id"])

    def finish(self) -> dict:
        if self.status == "completed":
            return {"records": len(self.seen), "path": str(self.output)}
        if self.status != "open":
            raise SubmissionError("writer is not open")
        if self.seen != self.expected:
            raise SubmissionError(f"incomplete submission: {len(self.expected - self.seen)} missing IDs")
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            os.link(self._temp, self.output)  # atomic creation; never replaces prior output
        except BaseException:
            self.abort()
            raise
        self.status = "completed"
        self._temp.unlink()
        return {"records": len(self.seen), "path": str(self.output)}

    def abort(self):
        if self.status == "completed":
            return
        self._stream.close()
        self._temp.unlink(missing_ok=True)
        self.status = "aborted"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
        elif self.status == "open":
            try:
                self.finish()
            except BaseException:
                self.abort()
                raise


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SubmissionError("duplicate JSON key")
        value[key] = item
    return value


def _constant(_value):
    raise SubmissionError("non-finite JSON number")


def validate_submission(path: str | Path, expected_ids: Iterable[str]) -> dict:
    """Read back strict JSONL; validate structure/ID coverage, not tool schemas or score."""
    expected, seen = _ids(expected_ids), set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                if not line.strip():
                    raise SubmissionError("blank JSONL line")
                record = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
                _record(record, expected, seen)
                seen.add(record["id"])
            except (ValueError, RecursionError) as exc:
                raise SubmissionError(f"invalid submission at line {number}") from exc
    if seen != expected:
        raise SubmissionError(f"incomplete submission: {len(expected - seen)} missing IDs")
    return {"records": len(seen), "coverage_complete": True}
