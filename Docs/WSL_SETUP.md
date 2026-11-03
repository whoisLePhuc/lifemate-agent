# LifeMate on WSL2

## Location and environments

Migrated from `D:\Projects\lifemate-workspace` to `/home/rnd/projects/lifemate-workspace` on 2026-11-03 (Asia/Bangkok). The original Windows workspace remains available as a backup. Workspace and agent retain separate Git histories.

From Ubuntu-22.04 as `rnd`:

```bash
cd ~/projects/lifemate-workspace/lifemate-agent
source .venv-training/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m lifemate.cli --help
```

`.venv` contains the baseline; `.venv-training` adds PEFT/bitsandbytes and inherits baseline dependencies through `baseline-dependencies.pth`. Both install the local project independently so editable-import hooks work in each environment. No Windows environment was reused. Python 3.12.15, torch 2.7.1+cu128, Transformers 4.56.2, Accelerate 1.10.1, PEFT 0.17.1 and bitsandbytes 0.48.1.

Rebuild in the same location with `uv` already available:

```bash
bash scripts/setup_wsl_env.sh
```

The setup uses the verified Linux baseline lock when present. Linux freeze files record installed dependencies separately from the historical Windows locks. To restore a separate, fully self-contained training environment instead of the overlay, create a new Python 3.12 venv and install `requirements-baseline.txt` together with `requirements-training-linux-py312.lock.txt`, then install this repo. Do not reuse an overlay after moving its parent; rerun setup to refresh absolute paths.

## Checks

```bash
LIFEMATE_DATA_ROOT=../Data \
LIFEMATE_MODEL_PATH=../Models/Qwen3-4B-Instruct-2507 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-training/bin/python -m unittest discover -s tests -v
.venv-training/bin/python -m pip check
```

Use Python pip for dependency inspection in the overlay: `uv pip check/freeze` does not account for inherited packages supplied through the custom `.pth` file.

Migration evidence, logs and smoke submissions are under `../Outputs/013-wsl-migration/run-20261103-01/`. Read `migration-report.json` for the final verification status. Raw outputs and weights remain outside Git. The smoke uses the first three sorted inner-validation IDs, never outer dev; it is an integration check, not a quality or speed benchmark.

## Git and file access

Local settings `core.autocrlf=input` and `core.filemode=false` keep the copied Windows checkout from appearing entirely modified. File contents were preserved; new shell scripts use LF. Access from Windows through `\\wsl.localhost\Ubuntu-22.04\home\rnd\projects\lifemate-workspace`. Run Python and Git in Ubuntu.

No vLLM backend, additional training or full validation run is introduced by this migration. Benchmark those separately after the existing backend is verified.
