"""Shared input/reference and controller result types."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AgentInput:
    id: str
    audio: Path
    context: dict
    history: list[dict]
    tools: list[dict]


@dataclass(frozen=True)
class Reference:
    audio_text: str
    history_audio_text: tuple[str, ...]
    tool_calls: list[dict]
    tool_results: list
    final_state: dict


@dataclass(frozen=True)
class TrainingExample:
    agent_input: AgentInput
    reference: Reference



@dataclass
class Outcome:
    stop: str
    elapsed_seconds: float
    decisions: int
    errors: int
    calls: int
