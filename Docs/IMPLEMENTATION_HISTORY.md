# Lịch sử tích hợp các thử nghiệm 001–005

Các thử nghiệm này đã hoàn tất tích hợp. Năm thư mục tương ứng trong workspace `Experiments` đã được xóa để không giữ runner, tests hoặc bản sao source song song. Mã hiện hành chỉ ở repo agent; dữ liệu/model/kết quả chi tiết vẫn ở workspace ngoài Git.

## Chức năng và bằng chứng đã giữ

Các đường dẫn Outputs bên dưới tính từ workspace root, không phải thư mục Docs.

| Giai đoạn | Mã và hướng dẫn hiện hành | Bằng chứng lịch sử |
|---|---|---|
| 001 – Data Loader | `lifemate/input/adapter.py`, [Data Loader](DATA_LOADER.md) | 1.545 train, 634 public đọc thành công; `Outputs/001-data-loader/run-20261102-02-integrated/report.json` |
| 002 – ToolEnv Adapter | `lifemate/tools/env_adapter.py`, [Adapter](TOOLENV_ADAPTER.md) | 1.545 ledger/results matches, 3.541 calls; `Outputs/002-toolenv-adapter/run-20261102-02-integrated/` |
| 003 – Replay và Writer | `evaluation/replay*.py`, `lifemate/output/writer.py`, [Replay](REPLAY_SUBMISSION.md) | 1.545 train full-state matches; CLI BTC 1.545 dòng/0 schema errors; `Outputs/003-replay-submission/run-20261102-02-integrated/` và `verification-20261102/` |
| 004 – Audit và split | `training/preparation.py`, `training/prepare_data.py`, [Chuẩn bị dữ liệu](DATA_PREPARATION.md) | 5.036 WAV train, 1.933 WAV public; split 1.236/309, seed 42; `Outputs/004-train-dev-split/run-20261102-01/` |
| 005 – Text baseline | `lifemate/agent`, `evaluation/run_text_baseline.py`, `evaluation/analysis.py`, [Baseline](TEXT_BASELINE.md) | Run 04 đủ 309 dev; 109 hash/82 raw-state/30 exact full-call matches; `Outputs/005-text-baseline/run-20261102-04-full-dev/` |

Các phần source runtime ở bảng nằm dưới `src/`. Tests hiện hành nằm dưới `tests/unit`, `tests/contract`, `tests/integration`. Đối chiếu AST của từng test trước khi xóa xác nhận **41/41 tests thử nghiệm đã được giữ trong repo agent**; suite agent tổng 49 tests (thêm 6 analysis và 2 structure tests).

## Cách chạy hiện tại

Từ root repo `lifemate-agent`, trong môi trường đã chuẩn bị:

```powershell
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"
.venv/Scripts/python.exe scripts/check_data.py --data-root ../Data --output ../Outputs/data-check/run-new/report.json
.venv/Scripts/python.exe scripts/check_adapter.py --data-root ../Data --split train --limit 3 --output ../Outputs/adapter-check/run-new/report.json
.venv/Scripts/python.exe -m lifemate.cli replay-train --data-root ../Data --output-dir ../Outputs/replay-check/run-new
.venv/Scripts/python.exe -m lifemate.cli prepare-data --data-root ../Data --output-dir ../Outputs/data-preparation/run-new --seed 42 --dev-fraction 0.2
.venv/Scripts/python.exe -m lifemate.cli text-baseline --data-root ../Data --manifest ../Outputs/004-train-dev-split/run-20261102-01/split_manifest.json --model-path ../Models/Qwen3-4B-Instruct-2507 --output-dir ../Outputs/text-baseline/run-new --limit 3
.venv/Scripts/python.exe -m lifemate.cli analyze-run --data-root ../Data --run-dir ../Outputs/005-text-baseline/run-20261102-04-full-dev --smoke-report ../Outputs/005-text-baseline/run-20261102-01-smoke/report.json --verify-gold-hashes --output-name analysis-rerun.json
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Chọn output mới cho mỗi lần chạy. Bật `LIFEMATE_DATA_ROOT` nếu muốn chạy test runtime opt-in. Audit audio/split có thể tốn thời gian; không cần chạy lại LLM để kiểm tra việc dọn thư mục. Không ghi đè dev manifest đã dùng làm mốc và không thay đổi raw data/toolkit.

## Khôi phục lịch sử khi cần

Workspace commit `fc61bb4` đã lưu thử nghiệm 005 và các giai đoạn trước; agent commit `996f13e` là mốc tích hợp trước refactor. Có thể đọc source bằng Git show hoặc checkout/worktree riêng để nghiên cứu lịch sử, không phục hồi mã cũ vào main để chạy luồng hiện hành.

Baseline run 04 được thực hiện trước commit, nên source_commit lịch sử vẫn null. Các commit nêu trên là mốc lưu trữ sau chạy, không phải bằng chứng SHA lúc inference. [Tóm tắt aggregate](Results/text-baseline-v0.1.json) và báo cáo gốc không bị sửa trong lần dọn này.

## Ghi chú thiết lập được giữ

Run 01 là finish-only smoke; run 02 bị dừng do áp lực VRAM; run 03 thử 3 mẫu chunked prefill; run 04 là baseline đủ 309 mẫu. Công cụ tải range trong thử nghiệm 005 đã gặp timeout khi tải CUDA wheel và không cần cho luồng chạy chính; không chuyển công cụ đó vào repo agent. Người dùng đã cài dependencies và CUDA được xác minh. Download model phát triển hiện dùng `scripts/download_baseline_model.py`.

49 tests và các workflow đã được xác minh trong các giai đoạn trước; bằng chứng dọn/xóa thử nghiệm được ghi riêng tại `Outputs/retire-experiments/run-20261102-01/`. Không coi smoke hoặc việc dọn source là một lần đánh giá chất lượng model mới.

## Planner Qwen3.5 và kiểm chứng run (tích hợp 05/11/2026)

- `lifemate.agent.qwen35_planner.Qwen35Planner` (từ thử nghiệm 015): dùng lại `LocalPlanner`, ép `enable_thinking=False` ở mọi lần render chat template, đặt EOS `<|im_end|>`/`<|endoftext|>` và ghi số tham số của checkpoint (từ chối nếu ≥ 5B). Cần `transformers>=5`; môi trường baseline cũ không import được module này. Tests: `tests/unit/test_qwen35_planner.py` (stub, không nạp model).
- `evaluation.verify_run.verify_run` (từ thử nghiệm 019): audit lại trace, đối chiếu hash replay với submission, tính lại điểm từng mẫu và tóm tắt run. Tests: `tests/unit/test_verify_run.py` (run tổng hợp, gồm các trường hợp phải bị từ chối). Trên 3 run thật của pipeline ASR+LLM, kết quả giống hệt bản gốc.
- 95 unit, 11 contract (2 skip), 14 integration test đạt. Pipeline sử dụng hai module này: `Experiments/023-asr-llm-pipeline/` trong workspace.
