"""Audio -> ASR -> LLM agent -> ToolEnv -> submission.

Three entry points, each a `lifemate` sub-command:
  transcribe  stage 1: audio to text (needs the `asr` extra; usually its own environment)
  run-agent   stage 2: transcripts to tool calls and submission (needs the Qwen3.5 stack)
  predict     both stages plus the organiser validator, in separate interpreters
"""
