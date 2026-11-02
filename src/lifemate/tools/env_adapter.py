"""One process uses one split runtime; one sample gets one ToolEnv attempt.

The official runtime is the sole authority for validation, dispatch, call ledger
and final hash. This adapter never exposes get_state, reset, or replay.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import re
import sys
from threading import RLock
from typing import Callable


class AdapterError(ValueError):
    """Invalid adapter configuration or input."""


class LifecycleError(AdapterError):
    """Operation is incompatible with the current session lifecycle."""


class RuntimeFailure(RuntimeError):
    """Runtime failed unexpectedly; the session must not be retried."""


_LOCK = RLock()
_ATTEMPTED: set[tuple[str, str, str]] = set()


def _runtime_factory(data_root: Path, split: str) -> Callable:
    package = (data_root / split / "zaic_tools").resolve()
    if not package.is_relative_to(data_root) or not (package / "__init__.py").is_file():
        raise AdapterError("missing or invalid split runtime")
    # Canonical import is required by some runtime imports. Never unload/reload
    # zaic_tools to switch splits: that can mix cached app modules.
    with _LOCK:
        for name, module in list(sys.modules.items()):
            if name == "zaic_tools" or name.startswith("zaic_tools."):
                source = getattr(module, "__file__", None)
                if source is None or not Path(source).resolve().is_relative_to(package):
                    raise LifecycleError("another zaic_tools runtime is loaded; use a separate process")
        sys.path.insert(0, str(package.parent))
        try:
            module = importlib.import_module("zaic_tools")
        finally:
            sys.path.pop(0)
        if Path(module.__file__).resolve() != package / "__init__.py":
            raise LifecycleError("loaded runtime does not match the selected split")
        return module.ToolEnv


def _arguments(name, arguments):
    if not isinstance(name, str) or not name:
        raise AdapterError("tool name must be a nonempty string")
    if not isinstance(arguments, dict):
        raise AdapterError("arguments must be an object")
    def string_keys(value):
        if isinstance(value, dict):
            return all(isinstance(k, str) and string_keys(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return all(string_keys(v) for v in value)
        return True
    if not string_keys(arguments):
        raise AdapterError("argument object keys must be strings")
    try:
        # Reject non-JSON values and NaN before dispatch; detach caller objects.
        return json.loads(json.dumps(arguments, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise AdapterError("arguments must contain finite JSON values") from exc


class ToolEnvAdapter:
    """A sample session; construction opens ToolEnv exactly once.

    validate() does not execute. call() dispatches directly, including calls the
    official validator may reject; those calls remain in the official ledger.
    finalize() freezes a cached submission record and prevents further calls.
    close() releases our handle only: this kit has no public close API.

    toolenv_factory is a test seam. Production callers must leave it unset.
    Duplicate attempts are blocked across adapters for the same resolved root,
    split and ID in this Python process, including failed construction attempts.
    Runner coordination is still required across processes or program restarts.
    """

    def __init__(self, data_root: str | Path, split: str, sample_id: str, *,
                 toolenv_factory: Callable | None = None):
        if split not in {"train", "public_test", "private_test"}:
            raise AdapterError("unsupported split")
        if (not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id)):
            raise AdapterError("invalid sample id")
        self.data_root = Path(data_root).resolve()
        self.split = split
        self.sample_id = sample_id
        self._status = "opening"
        self._record = None
        self._session = None
        self._lock = RLock()
        key = (str(self.data_root).casefold() if sys.platform == "win32" else str(self.data_root),
               split, sample_id)
        with _LOCK:
            if key in _ATTEMPTED:
                raise LifecycleError("sample already attempted in this process")
            factory = toolenv_factory if toolenv_factory is not None else _runtime_factory(self.data_root, split)
            _ATTEMPTED.add(key)
            try:
                self._session = factory(split, sample_id, root=self.data_root)
            except Exception as exc:
                self._status = "broken"
                raise RuntimeFailure("ToolEnv construction failed; do not retry this sample") from exc
        self._status = "active"

    @property
    def status(self) -> str:
        return self._status

    def _active(self):
        if self._status != "active":
            raise LifecycleError(f"session is {self._status}")

    def _invoke(self, method, *args):
        try:
            return getattr(self._session, method)(*args)
        except Exception as exc:
            self._status = "broken"
            raise RuntimeFailure("ToolEnv operation failed; session is unusable, no automatic retry") from exc

    def validate(self, name: str, arguments: dict) -> dict:
        """Official schema check only. No call is appended to the ledger."""
        with self._lock:
            self._active()
            args = _arguments(name, arguments)
            result = self._invoke("validate", name, args)
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                self._status = "broken"
                raise RuntimeFailure("invalid validation response")
            return deepcopy(result)

    def call(self, name: str, arguments: dict) -> dict:
        """Dispatch once. ok=True can still mean done=False; inspect payload."""
        with self._lock:
            self._active()
            args = _arguments(name, arguments)
            result = self._invoke("call", name, args)
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                self._status = "broken"
                raise RuntimeFailure("invalid tool response; execution may have occurred")
            return deepcopy(result)

    def get_tool_calls(self) -> list[dict]:
        with self._lock:
            if self._record is not None:
                return deepcopy(self._record["tool_calls"])
            self._active()
            return self._read_calls()

    def _read_calls(self):
        calls = self._invoke("get_tool_calls")
        if not isinstance(calls, list) or any(
            not isinstance(c, dict) or not isinstance(c.get("name"), str)
            or not isinstance(c.get("arguments"), dict) for c in calls
        ):
            self._status = "broken"
            raise RuntimeFailure("invalid runtime call ledger")
        return deepcopy(calls)

    def finalize(self) -> dict:
        """Return id, executed tool_calls and runtime hash; cached and detached."""
        with self._lock:
            if self._record is not None:
                return deepcopy(self._record)
            self._active()
            calls = self._read_calls()
            state_hash = self._invoke("get_state_hash")
            if not isinstance(state_hash, str) or not state_hash:
                self._status = "broken"
                raise RuntimeFailure("invalid runtime state hash")
            self._record = {"id": self.sample_id, "tool_calls": calls, "state_hash": state_hash}
            self._status = "finalized"
            return deepcopy(self._record)

    def close(self):
        """Discard handle; never reset the official environment."""
        with self._lock:
            self._session = None
            self._status = "closed"

    def __enter__(self):
        self._active()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
