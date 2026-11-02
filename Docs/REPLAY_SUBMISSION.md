# Replay train và Submission Writer

> Các đường dẫn `../Outputs/` và `../Experiments/` ghi trong nội dung dưới đây được tính từ repo `lifemate-agent`, không từ thư mục Docs. Chúng là bằng chứng workspace ngoài Git; lệnh triển khai nằm trong repo này.

## Hai chức năng riêng

`lifemate.output.writer.SubmissionWriter` dùng cho cả pipeline inference sau này: nhận record `{id, tool_calls, state_hash}` từ Adapter và ghi JSONL, không chọn tool, sửa arguments hoặc tính hash.

`evaluation.replay.TrainReplaySession` và `scripts/replay_train.py` chỉ dành cho kiểm tra nhãn train. Chúng chạy gold calls, đối chiếu kết quả và trạng thái cuối, rồi tạo file **`gold_replay_submission.jsonl`**. Đây là đáp án train được replay, không phải dự đoán của agent và không dùng để nộp public/private test.

## Submission Writer

```python
from lifemate.output.writer import SubmissionWriter, validate_submission

# expected_ids lấy từ đúng split bằng Data Loader, không lấy từ nhãn.
with SubmissionWriter(output_path, expected_ids) as writer:
    for record in finalized_records_from_adapter:
        writer.write(record)

summary = validate_submission(output_path, expected_ids)
```

- Chỉ nhận đúng ba trường `id`, `tool_calls`, `state_hash`.
- ID phải thuộc tập dự kiến, không trùng và đủ toàn bộ khi hoàn tất. Không bắt buộc thứ tự dòng theo ID.
- Call phải có đúng `name`, `arguments`; giữ nguyên thứ tự, call lặp và call runtime từ chối.
- Hash phải là SHA-256 hex chữ thường 64 ký tự theo kit hiện tại. Đây chỉ là kiểm tra hình dạng; nguồn hash phải từ Adapter, Writer không chứng minh tính đúng của hash.
- Chặn NaN/Infinity, kiểu không thuộc JSON, key không phải string và field thừa.
- UTF-8, mỗi record một dòng, newline trong text được JSON escape.
- Ghi file tạm cạnh file đích, flush/fsync rồi publish bằng hard link nguyên tử không ghi đè. Chỉ có tên file cuối sau khi đủ ID.
- Nếu thiếu mẫu, exception hoặc lỗi ghi/publish, không để lại file đích có vẻ hoàn chỉnh; context manager dọn file tạm. Nếu process bị kill đột ngột có thể còn file tạm, không dùng nó để nộp.
- File đích đã có sẽ bị từ chối. Chọn thư mục run mới thay vì ghi đè.

Writer cần filesystem hỗ trợ hard link (đã kiểm tra trên ổ D hiện tại). Nếu mount không hỗ trợ, publish fail rõ ràng; chưa có fallback ghi file không nguyên tử. Atomic publication không phải cam kết chống mất dữ liệu khi mất điện trên mọi filesystem. Không hỗ trợ resume/concurrent writers vào một output; người chạy phải quản lý run riêng.

`validate_submission(path, expected_ids)` đọc lại file và kiểm tra cấu trúc/ID. Nó không chạy tool, không kiểm tra ngữ nghĩa arguments hay tính điểm thi. Writer cũng không phân biệt record do model, gold replay hay dữ liệu tự tạo: caller chịu trách nhiệm chỉ truyền record từ phiên thực thi hợp lệ.

## Replay và đối chiếu trạng thái

Evaluator mở một ToolEnv train/mẫu thông qua Adapter. Nó lấy snapshot ban đầu bằng **accessor chính thức `get_state()`**, chỉ trong evaluator; không đọc file state trực tiếp. Sau các gold call của lượt cuối:

```text
expected_full_state = copy(initial_state)
expected_full_state.update(label.final_state)
full_state_match = actual_state == expected_full_state
```

Vì `label.final_state` chứa toàn bộ state của từng app thay đổi, phép so này kiểm tra cả app được ghi trong label và app phải giữ nguyên. Không replay call trong history. Kết quả snapshot không được trả cho agent hoặc dump vào báo cáo.

Evaluator dùng một subclass riêng cho train; Adapter inference không được thêm `get_state()` hoặc quyền đọc snapshot. Đây là ranh giới thiết kế, không phải cơ chế bảo mật ở cấp hệ điều hành: không import evaluator vào đường agent.

Report kiểm tra ledger, exact tool results, full state và số label-app khớp; report không chứa dữ liệu app, chỉ có ID và cờ mismatch nếu có. Submission replay chứa arguments thật nên giữ riêng tại Outputs, không commit hoặc chia sẻ ra ngoài.

Hash ghi trong file vẫn lấy từ `env.finalize()`. Không tự hash partial label. Chưa có gold hash độc lập hoặc scorer chính thức trong pipeline này, nên không báo điểm thi hay coi format hash là hash-match với gold.

## Chạy lại (PowerShell, từ repo chính)

```powershell
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"
python -m unittest discover -s tests -v
python scripts/replay_train.py --data-root 'D:\Projects\lifemate-workspace\Data' --output-dir 'D:\Projects\lifemate-workspace\Outputs\003-replay-submission\run-new'
```

`--output-dir` phải chưa tồn tại và nằm ngoài Data. Script không có tùy chọn test split để tránh replay nhãn ngoài train. Mỗi lần chạy là một tiến trình mới; không tạo môi trường thứ hai cho cùng mẫu trong lần chạy.

Kiểm tra schema bằng CLI BTC trong tiến trình riêng:

```powershell
$env:PYTHONPATH = 'D:\Projects\lifemate-workspace\Data\train'
python -m zaic_tools --split train --root 'D:\Projects\lifemate-workspace\Data' --validate 'D:\Projects\lifemate-workspace\Outputs\003-replay-submission\run-new\gold_replay_submission.jsonl'
```

CLI BTC tự mở một phiên/mẫu chỉ để kiểm tra allowlist/schema trong run riêng. Không chạy nó bên trong vòng inference đã mở ToolEnv. CLI này chưa phải scorer; xem số `schema errors` vì exit code của kit không tự bảo đảm không có lỗi schema.

## Bằng chứng thử nghiệm

Python 3.13.12, kit v3.2.0, schema 2.7.0. Thử nghiệm nguồn: `../Experiments/003-replay-submission` trong workspace.

- 8 tests mới: round-trip tiếng Việt/newline, bảo toàn call, thiếu/trùng/sai ID, cấu trúc/hash/JSON, không ghi đè kể cả race publish, abort khi lỗi, phát hiện file hỏng, đối chiếu app thay đổi và không thay đổi.
- Replay ban đầu: 1.545/1.545 mẫu khớp ledger, results và full state; 3.541 calls; 1.535/1.535 label-app khớp; 0 schema rejection.
- Đọc lại file kết quả đủ 1.545 ID duy nhất, đúng cấu trúc.
- Báo cáo ban đầu: `../Outputs/003-replay-submission/run-20261102-01/report.json`.
- Bản tích hợp đạt 28/28 tests; chạy lại train vẫn khớp 1.545/1.545 full states, results và ledgers. Báo cáo tại `../Outputs/003-replay-submission/run-20261102-02-integrated/report.json`.
- CLI chính thức xác nhận `1545 lines checked, 0 schema errors`. Log tests và CLI tại `../Outputs/003-replay-submission/verification-20261102/{tests.txt,official-validation.txt}`.

Đây là kiểm tra tích hợp bằng gold, không đo chất lượng LLM/ASR. Chưa chia train/dev hoặc chạy agent tự quyết định.
