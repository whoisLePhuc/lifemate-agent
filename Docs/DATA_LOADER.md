# Data Loader

> Các đường dẫn `../Outputs/` ghi trong nội dung dưới đây được tính từ repo `lifemate-agent`, không từ thư mục Docs. Chúng là bằng chứng workspace ngoài Git; lệnh triển khai nằm trong repo này.

Read-only loader for the organisers' JSONL and catalogs. No external runtime dependencies. Tested with Python 3.13.12; source targets Python 3.10+. The loader does not import ToolEnv or open environment files.

## Run from this repository (PowerShell)

```powershell
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"
python -m unittest discover -s tests -v
python scripts/check_data.py --data-root 'D:\Projects\lifemate-workspace\Data' --output 'D:\Projects\lifemate-workspace\Outputs\001-data-loader\run-new\report.json'
```

Choose a fresh output directory per run. `check_data.py` refuses to overwrite a report. Alternatively install the package with `python -m pip install -e .` in a development environment; the build backend must be available. Source-path execution above needs no installation or network access.

## Use

```python
from lifemate.input.adapter import DatasetLoader

loader = DatasetLoader(data_root, "train")
for item in loader.iter_inputs():
    # id: str; audio: absolute Path; context: dict; history: list; tools: list
    process_agent_input(item)

# TRAINING/EVALUATION ONLY; never pass reference to inference.
for example in loader.iter_training():
    train_or_evaluate(example.agent_input, example.reference)
```

`iter_inputs()` strips current/user-history `audio_text`, top-level labels and unknown history fields. It preserves supplied assistant text, calls and observations as conversation context. It never replays them. `Reference` contains `audio_text`, ordered `history_audio_text`, `tool_calls`, `tool_results`, and `final_state`. Reference access is train-only.

Tool schemas are filtered by the sample's installed-app allowlist, with `sys.ask_missing_info` added. JSON Schema extensions are preserved verbatim. Returned dictionaries are independent copies, not deeply immutable objects. Metadata is streamed; duplicate IDs, malformed JSON/keys, invalid installed tools, missing audio and audio path escapes produce `DataError`. The loader checks audio existence by default; `check_audio=False` skips existence checks, not path validation.

This is a metadata contract layer, not a full tool-argument validator, ASR system, date parser or scorer. It does not inspect audio content. `private_test` is a supported split name but has not been verified against a released private dataset. Error messages identify split/line without printing the record; supplied profile and observation content still requires private handling.

## Verification and provenance

Source experiment: [lịch sử triển khai](IMPLEMENTATION_HISTORY.md) in the research workspace. Initial real-data report: `../Outputs/001-data-loader/run-20261102-01/report.json`. Organiser kit v3.2.0 / catalog 2.7.0; 1,545 train and 634 public records passed metadata validation. Ten synthetic contract tests passed before integration. No gold replay, ToolEnv lifecycle or model quality has been verified by these checks.

The integrated source also passed all ten tests and the complete two-split check. Report: `../Outputs/001-data-loader/run-20261102-02-integrated/report.json`; test log: `../Outputs/001-data-loader/run-20261102-01/integrated-tests.txt`.
