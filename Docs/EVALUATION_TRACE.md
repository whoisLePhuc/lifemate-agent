# Evaluation & Trace v2

Triển khai ngày 02/11/2026. Mục tiêu: đo được chất lượng và giải thích lỗi của text agent trước khi đổi protocol hoặc fine-tune. Không thay SYSTEM prompt, JSON action contract, model weights, greedy decoding hoặc các budgets baseline.

## 1. Thành phần

| Thành phần | Vị trí | Trách nhiệm |
|---|---|---|
| Scorer local | `evaluation/scoring.py` | Schema-aware fuzzy matching, ghép call một-một, hash gate |
| Replay worker | `evaluation/replay_predictions.py` | Thực thi lại calls đúng thứ tự bằng toolkit BTC |
| Evaluation CLI | `evaluation/evaluate_run.py` | Chạy prediction/gold workers trong hai process, kiểm tra fingerprint rồi chấm |
| Trace | `src/lifemate/observability/trace.py` | JSONL append-only, flush/fsync từng event, audit với submission ledger |
| Provenance | `src/lifemate/observability/provenance.py` | Git/source fingerprints, config, dependency versions, model/tokenizer metadata |
| Instrumentation | Planner, controller, adapter và text runner hiện có | Token counts, finish reasons, raw output, validation/observation và stage timing |

Không thêm dependency. Scorer/replay/trace dùng stdlib; chạy LLM vẫn dùng môi trường baseline hiện có.

Prototype scorer và trace được kiểm thử trước trong experiment 006: 5 tests đạt, sau đó chuyển vào repo chính và mở rộng kiểm thử. Không giữ bản sao implementation trong workspace.

## 2. Scorer: cách tính và giới hạn

Tên/version: **`lifemate-local-fuzzy-v1`**. Mọi kết quả có `official_score=false`.

Theo [mô tả đánh giá BTC](https://api-challenge.zalo.ai/fe/api/config-detail?name=evaluationMetric&title=1&utm_source=default&utm_medium=default&utm_campaign=default), tool-name và non-fuzzy arguments cần giống nhau; fuzzy arguments cần token-F1 >0,6 sau lowercase/NFC; mỗi call chỉ ghép một lần và matching bỏ thứ tự.

Triển khai:

- Lấy `x-match: fuzzy` từ schema của tools được phép trong từng sample, không suy ra từ tên trường chung.
- Fuzzy dùng Unicode NFC rồi lowercase, giữ dấu tiếng Việt; token multiset và whitespace split. Punctuation được giữ nguyên.
- Argument keys phải bằng nhau; không tự thêm default. Arrays giữ thứ tự, xử lý schema `items` đệ quy. `true` không bằng số `1`; số JSON `1` và `1.0` được coi bằng nhau.
- Fuzzy cả hai chuỗi rỗng được quy ước F1=1; chỉ một rỗng là 0. Call bị schema reject khi replay không được ghép.
- Xây đồ thị tương thích và maximum cardinality bipartite matching, tránh thiếu cặp do ghép greedy.
- F1 = `2M / (số predicted + số gold)`. Mọi predicted call, kể cả rejected, nằm trong mẫu số.
- Gold không rỗng: hash sai thì điểm 0, hash đúng thì dùng F1. Gold rỗng: chỉ predicted rỗng mới được 1 theo trường hợp riêng trong mô tả BTC.

**Chưa được chứng nhận tương thích scorer BTC:** tài liệu chưa đủ rõ về tokenization, punctuation, chuỗi rỗng, key thiếu và một số cấu trúc lồng nhau. Các quy ước local đều nằm trong `CONVENTIONS` và được xuất vào report. Không dùng các hàm bỏ dấu/search token của toolkit để chấm fuzzy. Audit catalog train hiện tại tìm thấy 40 annotations fuzzy, tất cả kiểu string.

`matched_calls` là tổng cặp khớp trước hash gate. `mean_local_score` là trung bình điểm sau gate, không phải tỷ lệ hash match hay micro-F1 toàn dataset.

## 3. Chấm lại một submission dev

Chạy PowerShell từ repo:

```powershell
Set-Location D:\Projects\lifemate-workspace\lifemate-agent
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"

.venv/Scripts/python.exe -m lifemate.cli evaluate-run `
  --data-root ../Data `
  --submission ../Outputs/005-text-baseline/run-20261102-04-full-dev/submission.jsonl `
  --manifest ../Outputs/004-train-dev-split/run-20261102-01/split_manifest.json `
  --output-dir ../Outputs/006-evaluation-trace-v2/my-new-evaluation
```

Output directory phải mới và ngoài Data. Lệnh này chỉ đánh giá predictions có sẵn, không load/chạy LLM.

Nếu đánh giá một subset dev có chủ đích, thêm `--allow-subset`; mặc định có manifest sẽ yêu cầu đủ dev IDs. Không truyền manifest thì vẫn chỉ hỗ trợ các ID train có reference, không chấm nhãn public/private test.


Các file:

```text
my-new-evaluation/
  replay-predictions.json  # actual replay hash, ledger, schema/business failures
  replay-gold.json         # target hash từ gold replay độc lập
  scores.json              # local score + conventions + provenance + fingerprints
```

Hai workers chạy trong **hai process riêng**. Mỗi worker tạo đúng một ToolEnv mỗi ID. Không reset/retry và không tạo môi trường thứ hai trong cùng inference attempt. History không bị replay. Tất cả reads/writes app vẫn đi qua toolkit, không mở file môi trường trực tiếp.

Prediction replay gọi mọi submitted call, kể cả không hợp lệ; không pre-filter bằng validate. So ordered ledger thực tế và hash replay với submission. Có bất nhất thì evaluator thất bại và không xuất `scores.json` thành công; các file replay còn lại dùng để điều tra, không phải chứng nhận pass.

Scorer cũng dừng khi gold replay bị schema reject. Dữ liệu/catalog/toolkit fingerprints và submission hash phải giữ nguyên giữa hai workers.

## 4. Chỉ kiểm tra prediction replay

```powershell
.venv/Scripts/python.exe -m lifemate.cli replay-predictions `
  --data-root ../Data `
  --submission ../Outputs/005-text-baseline/run-20261102-04-full-dev/submission.jsonl `
  --output ../Outputs/006-evaluation-trace-v2/my-replay.json
```

Output phải là file chưa tồn tại. Exit code khác 0 nếu hash/ledger predictions không tái tạo được. `--mode gold` là worker evaluator lấy target hash, không phải chế độ inference và không làm predictions tốt hơn.

`schema_rejected_indices` và `business_failure_indices` là chỉ số call, bắt đầu từ 0. `ok=false` được ghi là schema rejection theo contract toolkit hiện tại; `ok=true, done=false` được ghi riêng là business failure. Không coi việc thiếu trường `done` trên một tool đọc là thất bại.

## 5. Trace cho các lượt text-baseline mới

Lệnh cũ vẫn hoạt động; trace được bật mặc định:

```powershell
.venv/Scripts/python.exe -m lifemate.cli text-baseline `
  --data-root ../Data `
  --manifest ../Outputs/004-train-dev-split/run-20261102-01/split_manifest.json `
  --model-path ../Models/Qwen3-4B-Instruct-2507 `
  --limit 3 `
  --output-dir ../Outputs/006-evaluation-trace-v2/my-new-gpu-trace
```

Bỏ `--limit` mới là chạy toàn bộ 309 dev. Không chạy full-dev chỉ để kiểm tra plumbing.

Các file mới ngoài report/submission/journal:

- `provenance.json`: source commit và dirty flag, diff/status/source-file hashes, config, package versions, model/tokenizer metadata fingerprints, revision nếu khôi phục được, GPU/CUDA.
- `trace.jsonl`: event tuần tự, UTC timestamp, thời gian tương đối, sample ID và decision index.
- `trace_audit.json`: kiểm tra run/sample boundaries, call start/result, ordered calls và record fingerprint đối chiếu submission.

Các event chính: `run_start`, `sample_start`, `decision_start`, `generation`, `parse_error` hoặc `parsed_action`, `validation`, `call_start`, `call_result` hoặc `call_error`, `controller_end`, `finalize`, `sample_end`, `run_end`.

Generation chứa prompt/new token count và nguyên nhân kết thúc `eos`, `token_limit`, `deadline`, `context_limit`, `backend_error`; khi có lỗi lưu phần output đã sinh. Parse error giữ raw output và lỗi parser riêng. Observation có flags schema rejection/business failure. Finalize tách thời gian đọc ledger và hash; report ghi thêm thời gian evaluation, finalize, write và publish submission.

Đây là **CPU wall-clock timings**, không thêm CUDA synchronize để đo kernel. Thời gian prefill/decode có thể chịu ảnh hưởng GPU async. Controller latency và total sample/evaluation latency được đặt tên riêng; chưa chứng nhận deadline audio hoặc tốc độ A30.

Theo dõi khi chạy:

```powershell
Get-Content ../Outputs/006-evaluation-trace-v2/my-new-gpu-trace/sample_results.jsonl -Tail 1
Get-Content ../Outputs/006-evaluation-trace-v2/my-new-gpu-trace/trace.jsonl -Tail 2
```

## 6. Crash, bảo mật và khả năng tái lập

- Flush/fsync mỗi event có overhead và nằm trong thời gian chạy. Không hứa trace luôn giữ nguyên quyết định ở sát deadline.
- `call_start` không có `call_result` là call có kết quả chưa xác định. Không tự gọi lại, không suy ra chưa áp dụng, không resume sample đó bằng ToolEnv mới.
- Audit nhận diện sample chưa hoàn tất, call treo, thiếu run_end, JSON bị cắt, sequence sai hoặc ledger/record fingerprint lệch. Trace không có chữ ký chống sửa đồng thời; nó là công cụ kiểm tra nhất quán, ToolEnv vẫn là nguồn ledger/hash.
- Trace có raw output và observations có thể chứa thông tin cá nhân; chỉ lưu trong Outputs ngoài Git. Không log gold, final_state hoặc evaluator state snapshot vào trace.
- Metadata revision lấy từ config/snapshot/local download metadata; không bịa revision khi thiếu. Metadata hash không thay thế chứng nhận toàn bộ weight shards.
- Khi source dirty, commit chỉ là nền; source/diff hashes mô tả code thay đổi. Không gán commit mới vào run 04 lịch sử.
- P0 chỉ làm lỗi repair dễ quan sát hơn; chưa đổi nội dung repair prompt hoặc policy agent để giữ đối chứng. Việc đó thuộc bước cải thiện sau.

## 7. Kết quả xác minh

### Submission baseline 309 mẫu — không chạy lại LLM

Output: `Outputs/006-evaluation-trace-v2/run-20261102-01-evaluate-baseline/`, tính từ workspace root.

| Chỉ số | Kết quả |
|---|---:|
| Prediction hashes tái tạo đúng | 309/309 |
| Ordered prediction ledgers khớp | 309/309 |
| Schema rejected calls | 0 |
| Business failure calls | 40 |
| Hash matches với gold | 109/309 |
| Local fuzzy matched calls, trước gate | 206 |
| Gold / predicted calls | 705 / 862 |
| Mean local score, sau gate | **0,1552088149** |
| Mẫu có local score >0 | 62/309 |

Không gọi 0,1552 là điểm thi chính thức. Nó giúp so sánh có kiểm soát dưới các quy ước đã ghi, khác cả 35,3% hash match và 9,7% exact full-call match.

Phân tích cũ được chạy lại trên bản sao report/submission ngoài baseline directory tại `run-20261102-03-legacy-analysis/`: JSON bằng hoàn toàn `analysis-hash-verified.json` cũ; 109 hash matches, 82 raw-state matches, 30 exact full-call matches vẫn giữ nguyên.

### Smoke LLM thật, 3 mẫu dev

Output: `Outputs/006-evaluation-trace-v2/run-20261102-02-gpu-trace/`.

- Qwen3-4B-Instruct-2507 BF16, cùng revision/seed/prompt/budget.
- 13 decisions, 9 actual calls, 80 trace events; cả 13 generations kết thúc bằng EOS.
- Trace audit đạt, không mất call hoặc record.
- **3/3 submission records bằng đúng baseline run 04** ở cùng IDs.
- Replay độc lập tại `run-20261102-04-evaluate-gpu-trace/` tái tạo đúng 3/3 hashes và ledgers.

Đây là kiểm tra tích hợp, không phải cải thiện chất lượng hoặc bằng chứng tương đương trên toàn dev. Ba mẫu có local score 0; P0 không được thiết kế để chữa hành vi agent.

### Tests

```powershell
$env:LIFEMATE_DATA_ROOT = 'D:\Projects\lifemate-workspace\Data'
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

62 tests đạt khi bật contract test với toolkit thật. Không đặt biến này: 61 đạt, 1 skipped. Các test mới kiểm tra matching tối đa, multiplicity, fuzzy threshold/NFC, JSON types, hash gate, schema rejection, ordered replay, provenance, trace interruption, không retry call chưa xác định và metadata EOS/token-limit/context/deadline trong vòng token bằng fake backend CPU.

Wheel cũng được build và giải nén ngoài source: ba CLI help hoạt động, đánh giá lại 3 mẫu bằng hai workers đạt, các module mới trong wheel bằng source. Khi wheel nằm trong Git workspace cha, provenance không nhận nhầm commit của workspace đó là commit của source agent. Bằng chứng build/tests ở `Outputs/006-evaluation-trace-v2/run-20261102-05-verification/`.

Bản aggregate đã bỏ ID/arguments/observations và đường dẫn máy: `Docs/Results/evaluation-trace-v2.json`. Kết quả đầy đủ nằm ngoài Git trong Outputs.

## 8. Bước tiếp theo

P0 đã có công cụ đo và bằng chứng kiểm chứng. Tiếp theo nên tạo experiment native Qwen tool protocol, giữ nguyên weights/transcript/budget và so paired trên tập chẩn đoán trước khi chạy lại 309 dev. Scorer vẫn cần đối chiếu chính thức khi BTC cung cấp đủ chi tiết.
