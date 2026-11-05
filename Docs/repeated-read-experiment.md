# H1 — repeated reads

The experimental `repeat_reads_v1` policy blocks a repeated successful read
with identical tool name and JSON arguments. It gives the model feedback and
keeps the original observation in the conversation. Each blocked action uses
the existing repair budget, and exhausting it stops with `repeat_read_limit`.
The trace records `repeated_read_blocked` without adding a ToolEnv call.

Any executed tool whose kind is not `read` invalidates all remembered reads.
Only observations with `ok=true` and without `done=false` are remembered.
This conservative rule allows retries after a failed or incomplete read and
allows reads after writes, including writes to another app.

Compare `--policy off` and `--policy repeat_reads_v1` with the same inputs,
checkpoint and budgets using `lifemate text-baseline` or `lifemate run-agent`.
For example, append `--policy repeat_reads_v1` to an existing run-agent
command. The default remains `off`. The policy currently supports
`custom_json` only.

Check local score, state hash match, repeated calls, stop reasons and latency
on paired runs. The automated tests use simulated tools and do not establish
an improvement on checkpoint 0880 or the competition evaluator.

Recovery variants use the same CLI and allow one model decision after a
blocked duplicate per sample, within the existing repair budget.

- `--policy repeat_reads_observation` (H1a) includes the earlier observation
  in the repair message, including empty lookup results.
- `--policy repeat_reads_options` (H1b) also suggests next steps and lists
  conditional clarification candidates for required or alternative fields
  linked to the observed collection by `x-ref`. The model still chooses the
  action and the controller validates every call before execution.
