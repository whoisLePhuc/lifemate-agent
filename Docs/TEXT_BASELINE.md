# Baseline LLM dùng transcript chuẩn

> Mã đã được tổ chức theo [kiến trúc mới](REFACTOR_ARCHITECTURE.md). Các lệnh script dưới đây vẫn tương thích; CLI thống nhất là `python -m lifemate.cli text-baseline ...`. Kết quả run 04 là baseline lịch sử, không được đo lại trong lần refactor.

> Các đường dẫn `../Outputs/` và `../Experiments/` ghi trong nội dung dưới đây được tính từ repo `lifemate-agent`, không từ thư mục Docs. Chúng là bằng chứng workspace ngoài Git; lệnh triển khai nằm trong repo này.

## Trạng thái hiện tại

Run `run-20261102-04-full-dev` đã hoàn tất toàn bộ **309 mẫu dev** với Qwen3-4B-Instruct-2507, chưa fine-tune LifeMate. Kết quả: **109 hash matches**, **82 raw-state matches**, **30 exact full-call matches**; đây chưa phải điểm thi chính thức.

- [Phân tích và giới hạn](BASELINE_ANALYSIS.md).
- [Tóm tắt aggregate để lưu Git](Results/text-baseline-v0.1.json): model revision, fingerprints, cấu hình, môi trường và số liệu; không chứa bản ghi từng mẫu.
- Model, dữ liệu, transcript và output đầy đủ ở workspace ngoài Git.

## Đường xử lý và ranh giới dữ liệu

Transcript chuẩn lượt cuối/user history → prompt → LLM → parse → validate → ToolEnv → observation → bước tiếp theo. Adapter cung cấp record cho Submission Writer; evaluator độc lập so state/hash/calls.

Planner nhận transcript string, context, history và schema tool khả dụng. Không nhận Reference, gold action/result/final_state hoặc snapshot app. History đã thực thi, không replay. Chỉ evaluator train dùng accessor chính thức để so trạng thái; không đọc file môi trường trực tiếp.

## Môi trường đã chạy

Windows, Python 3.12.15, RTX 4070 Ti SUPER 16 GB (khác A30 của BTC). Torch 2.7.1+cu128, Transformers 4.56.2, Accelerate 1.10.1, Hugging Face Hub 0.34.4. [Các phiên bản đầy đủ đã ghi nhận](../requirements-baseline-windows-py312.lock.txt) là môi trường Windows/Python 3.12 của run này, không phải lock Docker/A30 cuối cùng; chưa xác minh cài mới mọi phiên bản trên máy khác.

Từ repo `lifemate-agent`, tạo venv và cài bằng pip:

```powershell
uv venv --python 3.12 .venv
.venv/Scripts/python.exe -m ensurepip --upgrade
.venv/Scripts/python.exe -m pip install -r requirements-baseline.txt --timeout 600 --retries 10
```

Nếu muốn dùng uv, thay lệnh pip install bằng:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-baseline.txt
```

Không cần tạo lại venv nếu đã có. Để khôi phục đúng bộ phiên bản đã ghi nhận thay vì chỉ pin dependencies trực tiếp, dùng `-r requirements-baseline-windows-py312.lock.txt`. Đây là lệnh cài môi trường Windows CUDA; không mặc định dùng trên CPU/Linux hoặc thay máy chấm mà chưa xác minh.

## Model local và cấu hình

[Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f17c01a7cc42f958dc650907174af0554` (commit 17/09/2025), 4.022.468.096 tham số. Chỉ tải ở giai đoạn phát triển:

```powershell
.venv/Scripts/python.exe scripts/download_baseline_model.py --output-dir ../Models/Qwen3-4B-Instruct-2507
```

Inference dùng local_files_only, BF16 và SDPA; không API bên ngoài. Giữ trọng số/license ngoài Git. Không chạy script download trong inference. Trước nộp cần audit tổng tham số cả ASR, license và Docker riêng.

| Cấu hình baseline | Giá trị |
|---|---:|
| Seed, greedy decoding | 42, do_sample=False |
| Context gồm prompt và phần sinh | 16.384 token |
| Token sinh tối đa/action | 384 |
| Prefill chunk | 256 token |
| Quyết định/call/repair tối đa | 12 / 8 / 2 |
| Deadline controller | 30 s, soft |

Backend greedy forward giữ KV cache trong một quyết định, tạo cache mới cho quyết định tiếp theo; chỉ giữ logits token cuối. Không cắt prompt âm thầm. Budget soft được kiểm tra giữa các lần forward; chưa có watchdog cưỡng bức prefill/kernel/tool, nên chưa chứng minh trần thời gian toàn pipeline của BTC. Model load và evaluator không nằm trong controller latency.

## Chạy và phân tích lại

Từ repo chính, output mới cho mỗi run:

```powershell
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"
$env:HF_HUB_OFFLINE = '1'
.venv/Scripts/python.exe scripts/run_text_baseline.py --data-root ../Data --manifest ../Outputs/004-train-dev-split/run-20261102-01/split_manifest.json --model-path ../Models/Qwen3-4B-Instruct-2507 --output-dir ../Outputs/005-text-baseline/run-new --limit 3
```

Bỏ `--limit 3` để chạy đủ 309 dev. Tập con dùng ID dev sort đầu tiên, không chọn theo nhãn. Manifest phải khớp fingerprint nguồn; mỗi run một tiến trình, mỗi mẫu một ToolEnv, không reset/retry. Không chạy hai LLM run cùng lúc trên GPU này.

Thay `--model-path ...` bằng `--smoke-finish-only` để kiểm tra plumbing không cần model. Smoke luôn finish không phải kết quả LLM: tác vụ đọc/hỏi có thể giữ nguyên state dù hành vi sai.

Phân tích kết quả đã có, không chạy LLM:

```powershell
.venv/Scripts/python.exe scripts/analyze_run.py --data-root ../Data --run-dir ../Outputs/005-text-baseline/run-20261102-04-full-dev --smoke-report ../Outputs/005-text-baseline/run-20261102-01-smoke/report.json --verify-gold-hashes --output-name analysis-rerun.json
```

Bỏ `--verify-gold-hashes` nếu chỉ phân tích call/report. Khi bật, evaluator replay nhãn train trong run riêng, một ToolEnv/mẫu, không thay predictions. Chặn ID thiếu/trùng, thiếu references, output ngoài run hoặc ghi đè báo cáo.

## Output và đọc chỉ số

- `submission.jsonl`: calls/hash thực từ môi trường, đủ ID được chọn; không đưa vào Git.
- `report.json`: cấu hình nhận diện, stop reason và metrics từng mẫu. `full_state_match` là raw-state equality.
- `sample_results.jsonl`: runner hiện tại ghi ngay metrics/record từng mẫu. Run 04 bắt đầu trước thay đổi này nên **không có journal**; raw model outputs cũng không được lưu ở run 04.
- `analysis*.json`: exact unordered call diagnostics và optional canonical hash match.

`finished` chỉ là model dừng. Hash match không thay tool-call matching. Exact ordered/unordered equality chưa hỗ trợ fuzzy token-F1 và chưa là điểm BTC. Logs không dump prompt/transcript; journal/submission có arguments thật, cần giữ riêng. Trường hợp runtime không xác định làm run fail thay vì hash giả. Writer không publish output thiếu ID.

## Lịch sử thử nghiệm cần giữ

- Run 01: smoke finish-only, đủ 309 ID; xác minh plumbing, không phải model quality.
- Run 02: bị dừng vì prefill nguyên khối gây áp lực VRAM/timeout; không có report hoàn chỉnh.
- Run 03: chunked prefill, 3 mẫu thật, trung bình 14,73 s; chỉ là smoke GPU.
- Run 04: đủ 309 mẫu thật; controller mean 11,95 s, p95 29,70 s, max 29,78 s. Không tính model loading/evaluator. Kết quả và giới hạn trong bản tóm tắt Git.

Công cụ tải theo range chỉ là xử lý sự cố trong Experiments. Lượt tải CUDA theo range đã timeout; không còn tiến trình thiết lập chờ chạy. Thư viện sau đó được người dùng cài và CUDA đã xác minh. Backend đã kiểm tra prefill chia đoạn so nguyên khối trên Qwen3 nhỏ (max logit difference khoảng 5,96e-08); chưa khẳng định bitwise reproducible trên mọi GPU/kernel.
