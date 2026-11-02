"""Strict, streaming readers for the organisers' JSONL and tool catalogs.

Only known metadata/catalog files and audio paths are used. No runtime is imported
and no environment directory is enumerated, opened, or parsed.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator


class DataError(ValueError):
    """Invalid dataset contract. Messages intentionally omit sample contents."""


from lifemate.contracts import AgentInput, Reference, TrainingExample


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError("duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(_value):
    raise DataError("non-finite JSON number")


def _parse(text: str):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_bad_constant)


def _require(ok: bool, message: str):
    if not ok:
        raise DataError(message)


def _calls(value, field: str) -> list[dict]:
    _require(isinstance(value, list), f"{field} must be an array")
    for call in value:
        _require(isinstance(call, dict), f"{field} entry must be an object")
        _require(isinstance(call.get("name"), str) and bool(call["name"]),
                 f"{field} entry needs a tool name")
        _require(isinstance(call.get("arguments"), dict),
                 f"{field} arguments must be an object")
    # Whitelist the public tool-call fields, excluding arbitrary extra metadata.
    return [{"name": c["name"], "arguments": deepcopy(c["arguments"])} for c in value]


class DatasetLoader:
    """Load one split. Each iterator opens JSONL anew; records are streamed.

    iter_inputs() never returns gold/transcript fields. iter_training() is an
    explicit train-only entry point; do not pass its Reference to an agent.
    tools contains the full tool schemas allowed for the current sample plus
    sys.ask_missing_info. Returned mutable dictionaries are independent copies.
    """

    SPLITS = frozenset({"train", "public_test", "private_test"})

    def __init__(self, data_root: str | Path, split: str, *, check_audio: bool = True):
        _require(split in self.SPLITS, "unsupported split")
        self.data_root = Path(data_root).resolve()
        self.split = split
        self.split_root = (self.data_root / split).resolve()
        _require(self.split_root.is_relative_to(self.data_root), "split escapes data root")
        self.check_audio = check_audio
        self._jsonl = self._metadata_path(f"{split}.jsonl")
        catalog_path = self._metadata_path(f"{split}_tools_catalog.json")
        try:
            catalog = _parse(catalog_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise DataError("cannot read valid split catalog") from exc
        _require(isinstance(catalog, dict), "catalog must be an object")
        _require(isinstance(catalog.get("schema_version"), str),
                 "catalog needs schema_version")
        _require(isinstance(catalog.get("apps"), list), "catalog.apps must be an array")
        self.schema_version = catalog["schema_version"]
        self._apps = set()
        self._tools = {}
        for app in catalog["apps"]:
            _require(isinstance(app, dict) and isinstance(app.get("id"), str),
                     "invalid catalog app")
            app_id = app["id"]
            _require(app_id not in self._apps, "duplicate catalog app")
            self._apps.add(app_id)
            _require(isinstance(app.get("tools"), list), "catalog app needs tools array")
            for tool in app["tools"]:
                _require(isinstance(tool, dict), "invalid catalog tool")
                name = tool.get("name")
                _require(isinstance(name, str) and name.startswith(app_id + "."),
                         "catalog tool name/app mismatch")
                _require(name not in self._tools, "duplicate catalog tool")
                _require(tool.get("app") == app_id, "catalog tool owner mismatch")
                _require(isinstance(tool.get("parameters"), dict),
                         "catalog tool needs parameters object")
                self._tools[name] = deepcopy(tool)
        _require("sys.ask_missing_info" in self._tools, "missing system clarification tool")

    def _metadata_path(self, name: str) -> Path:
        path = (self.split_root / name).resolve()
        _require(path.parent == self.split_root, "metadata escapes split root")
        _require(not any(p.lower().endswith("_environment") for p in path.parts),
                 "environment paths are forbidden")
        _require(path.is_file(), "missing split metadata file")
        return path

    @property
    def catalog_counts(self) -> dict[str, int]:
        return {"apps": len(self._apps), "tools": len(self._tools)}

    def _audio_path(self, value) -> Path:
        _require(isinstance(value, str) and bool(value), "audio path must be a string")
        # Interpret both separator styles regardless of host OS.
        _require(not PureWindowsPath(value).drive and not PureWindowsPath(value).is_absolute(),
                 "absolute audio path is forbidden")
        rel = PurePosixPath(value.replace("\\", "/"))
        _require(not rel.is_absolute() and ".." not in rel.parts,
                 "audio path traversal is forbidden")
        _require(bool(rel.parts) and rel.parts[0] == "audio", "audio must be under audio/")
        _require(not any(p.lower().endswith("_environment") for p in rel.parts),
                 "environment paths are forbidden")
        path = self.split_root.joinpath(*rel.parts).resolve()
        audio_root = (self.split_root / "audio").resolve()
        _require(audio_root.is_relative_to(self.split_root), "audio directory escapes split")
        _require(path.is_relative_to(audio_root), "audio path escapes audio directory")
        _require(not any(p.lower().endswith("_environment") for p in path.parts),
                 "environment paths are forbidden")
        if self.check_audio:
            _require(path.is_file(), "missing audio file")
        return path

    def _active_tools(self, context: dict) -> list[dict]:
        installed = context.get("installed_apps")
        _require(isinstance(installed, dict), "context.installed_apps must be an object")
        names = []
        for app_id, config in installed.items():
            _require(app_id in self._apps, "installed app absent from catalog")
            _require(isinstance(config, dict) and isinstance(config.get("tools"), list),
                     "installed app needs tools array")
            for name in config["tools"]:
                _require(isinstance(name, str) and name in self._tools,
                         "installed tool absent from catalog")
                _require(self._tools[name]["app"] == app_id, "installed tool/app mismatch")
                _require(name not in names, "duplicate installed tool")
                names.append(name)
        if "sys.ask_missing_info" not in names:
            names.append("sys.ask_missing_info")
        return [deepcopy(self._tools[name]) for name in names]

    def _input(self, row: dict) -> AgentInput:
        sample_id = row.get("id")
        _require(isinstance(sample_id, str) and bool(sample_id.strip()), "id must be nonempty")
        raw_context = row.get("context")
        _require(isinstance(raw_context, dict), "context must be an object")
        for key in ("current_time", "timezone"):
            _require(isinstance(raw_context.get(key), str) and bool(raw_context[key]),
                     f"context.{key} must be a nonempty string")
        for key in ("location", "user_profile"):
            _require(isinstance(raw_context.get(key), dict), f"context.{key} must be an object")
        # Select only known context fields; never forward the raw row wholesale.
        context = {key: deepcopy(raw_context[key]) for key in (
            "current_time", "timezone", "location", "user_profile", "installed_apps")
            if key in raw_context}
        tools = self._active_tools(context)
        raw_history = row.get("history")
        _require(isinstance(raw_history, list), "history must be an array")
        history = []
        for turn in raw_history:
            _require(isinstance(turn, dict), "history turn must be an object")
            role = turn.get("role")
            if role == "user":
                # audio_text is deliberately omitted even on train.
                history.append({"role": "user", "audio": self._audio_path(turn.get("audio"))})
            elif role == "assistant":
                _require(isinstance(turn.get("text"), str), "assistant history needs text")
                clean = {"role": "assistant", "text": turn["text"]}
                if "tool_calls" in turn:
                    clean["tool_calls"] = _calls(turn["tool_calls"], "history.tool_calls")
                if "tool_results" in turn:
                    _require(isinstance(turn["tool_results"], list),
                             "history.tool_results must be an array")
                    clean["tool_results"] = deepcopy(turn["tool_results"])
                history.append(clean)
            else:
                raise DataError("unsupported history role")
        return AgentInput(sample_id, self._audio_path(row.get("audio")), context, history, tools)

    def _rows(self) -> Iterator[tuple[int, dict]]:
        seen = set()
        try:
            with self._jsonl.open(encoding="utf-8-sig") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = _parse(line)
                        _require(isinstance(row, dict), "record must be an object")
                        sample_id = row.get("id")
                        _require(isinstance(sample_id, str) and bool(sample_id.strip()),
                                 "id must be nonempty")
                        _require(sample_id not in seen, "duplicate sample id")
                        seen.add(sample_id)
                    except ValueError as exc:
                        raise DataError(f"{self.split} line {line_number}: invalid JSON/record") from exc
                    yield line_number, row
        except (OSError, UnicodeError) as exc:
            raise DataError(f"cannot read {self.split} JSONL") from exc

    def iter_inputs(self) -> Iterator[AgentInput]:
        """Default inference path, excluding current/history gold transcripts and label."""
        for number, row in self._rows():
            try:
                yield self._input(row)
            except DataError as exc:
                raise DataError(f"{self.split} line {number}: {exc}") from exc

    def iter_training(self) -> Iterator[TrainingExample]:
        """Explicit train/evaluation path; unavailable for either test split."""
        _require(self.split == "train", "references are available only for train")
        for number, row in self._rows():
            try:
                agent_input = self._input(row)
                _require(isinstance(row.get("audio_text"), str), "train needs audio_text")
                history_text = []
                for turn in row["history"]:
                    if turn["role"] == "user":
                        _require(isinstance(turn.get("audio_text"), str),
                                 "train user history needs audio_text")
                        history_text.append(turn["audio_text"])
                label = row.get("label")
                _require(isinstance(label, dict), "train needs label object")
                calls = _calls(label.get("tool_calls"), "label.tool_calls")
                results = label.get("tool_results")
                _require(isinstance(results, list) and len(results) == len(calls),
                         "label call/result lengths differ")
                _require(isinstance(label.get("final_state"), dict),
                         "label.final_state must be an object")
                reference = Reference(row["audio_text"], tuple(history_text), calls,
                                      deepcopy(results), deepcopy(label["final_state"]))
                yield TrainingExample(agent_input, reference)
            except DataError as exc:
                raise DataError(f"{self.split} line {number}: {exc}") from exc
