#!/usr/bin/env bash
# Rebuild Linux environments without changing the baseline dependency pins.
set -euo pipefail
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
uv python install 3.12
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 --seed .venv; fi
baseline_requirements=(-r requirements-baseline.txt)
if [[ -f requirements-baseline-linux-py312.lock.txt ]]; then
    baseline_requirements+=(-r requirements-baseline-linux-py312.lock.txt)
fi
uv pip install --python .venv/bin/python "${baseline_requirements[@]}"
uv pip install --python .venv/bin/python -e .
if [[ ! -x .venv-training/bin/python ]]; then uv venv --python .venv/bin/python --seed .venv-training; fi
baseline_site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
training_site=$(.venv-training/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$baseline_site" > "$training_site/baseline-dependencies.pth"
uv pip install --python .venv-training/bin/python --no-deps -r requirements-training.txt
uv pip install --python .venv-training/bin/python --no-deps -e .
printf 'Ready. Activate: source %s/.venv-training/bin/activate\n' "$repo"
