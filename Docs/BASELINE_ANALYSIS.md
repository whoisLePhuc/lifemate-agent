# Phân tích baseline text trên 309 mẫu dev

> Các đường dẫn `../Outputs/` ghi trong nội dung dưới đây được tính từ repo `lifemate-agent`, không từ thư mục Docs. Chúng là bằng chứng workspace ngoài Git; lệnh triển khai nằm trong repo này.

Ngày phân tích: 02/11/2026. Model Qwen3-4B-Instruct-2507, chưa fine-tune theo LifeMate; transcript chuẩn lượt cuối và user history, không có lỗi ASR trong đầu vào. Đây là đánh giá toàn pipeline prompt/planner/controller/backend, không tách riêng năng lực model.

## Nguồn và phương pháp

Run: `Outputs/005-text-baseline/run-20261102-04-full-dev/`, tính từ workspace root. `report.json` ghi 309 mẫu, `submission.jsonl` là call/hash thật. `analysis.json` là chẩn đoán ban đầu; `analysis-hash-verified.json` thêm đối chiếu hash đích qua replay gold train trong một lượt evaluator riêng (một ToolEnv/mẫu, không model inference, không thay dự đoán gốc, không đọc file môi trường trực tiếp).

Lệnh tái tạo từ repo `lifemate-agent`, dùng tên output mới. Công cụ nằm trong chính repo này; không cần mã của workspace Experiments:

```powershell
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"
.venv/Scripts/python.exe scripts/analyze_run.py --data-root ../Data --run-dir ../Outputs/005-text-baseline/run-20261102-04-full-dev --smoke-report ../Outputs/005-text-baseline/run-20261102-01-smoke/report.json --verify-gold-hashes --output-name analysis-rerun.json
```

Các phép so call trong phân tích dùng bằng nhau chính xác, có tính số lần lặp và bỏ thứ tự; chưa dùng fuzzy token-F1. Vì vậy không coi tỷ lệ exact-match hoặc các count ở đây là điểm thi BTC. Không truy cập dữ liệu test có nhãn.

[Bản tóm tắt chuẩn bị đưa vào Git](Results/text-baseline-v0.1.json) chứa số liệu tổng hợp, model revision, fingerprint, phiên bản thư viện và lệnh chạy; không chứa transcript, ID từng mẫu, arguments, trạng thái app hoặc đường dẫn máy cá nhân. Report/submission đầy đủ vẫn ở Outputs ngoài Git. Vì run được thực hiện trước commit, trường source_commit để null thay vì gán nhầm một SHA lịch sử.

Implementation tích hợp đã được kiểm tra lại trước commit: 6 tests phân tích mới đạt, suite tổng 47/47 đạt; `analysis-precommit-verified.json` bằng phân tích gốc. Xem [trạng thái chuẩn bị commit](PRECOMMIT_STATUS.md).

## Chỉ số chính và đính chính

| Chỉ số | Kết quả |
|---|---:|
| Đủ record/ID dev | 309/309 |
| Khớp toàn bộ state nguyên bản | 82/309 = 26,5% |
| Khớp hash đích từ ToolEnv BTC | 109/309 = 35,3% |
| Đủ cả bộ call và arguments chính xác, bỏ thứ tự | 30/309 = 9,7% |
| Đúng bộ tên tool và số lần xuất hiện, chưa xét arguments | 80/309 = 25,9% |
| Khớp hash ở mẫu thực sự có app thay đổi theo label | 63/227 = 27,8% |

Con số 82 ban đầu là phép so raw state, không phải canonical hash theo BTC. Hash bỏ một số trường như text tự do/app thông tin nên có thêm 27 mẫu hash khớp. Phép so raw state vẫn hữu ích cho debug, nhưng phải đặt tên rõ, không dùng thay tỷ lệ hash match khi báo cáo thi.

Hash đúng chưa đồng nghĩa làm đúng yêu cầu: các tác vụ đọc, hỏi thiếu và ngoài phạm vi có thể giữ nguyên state. Trong 109 mẫu hash khớp, chỉ 30 mẫu có toàn bộ call/arguments khớp chính xác; những mẫu còn lại cần chấm fuzzy/ghép call chính thức trước khi kết luận điểm.

Smoke luôn finish cũng có 82 raw-state matches. Hai tập đúng không giống nhau: model giữ đúng 45 mẫu cùng smoke, cải thiện 37 mẫu và làm sai 37 mẫu mà smoke giữ đúng. Không suy ra model không có năng lực chỉ vì hai tổng bằng nhau. Không so trực tiếp raw-state smoke với canonical-hash model.

## Hỏi thiếu thông tin là lỗi ưu tiên cao

Gold có 54 mẫu cần `sys.ask_missing_info`, model chỉ gọi tool này ở 12 mẫu. Chỉ 2 mẫu thuộc nhóm thật sự cần hỏi; 52/54 mẫu cần hỏi bị bỏ sót, 10/12 mẫu model hỏi không thuộc nhóm gold cần hỏi. Recall theo sự hiện diện tên tool là 3,7%, precision 16,7%; đây chưa phải matching arguments của tool hỏi.

Mẫu `train_0116`: gold chỉ hỏi thiếu thông tin, dự đoán gọi `phone.call` ba lần. Đây là dấu hiệu model không duy trì chính sách hỏi khi thiếu dữ kiện và còn lặp hành động. Chưa có raw output/observation trace của run này để xác định đầy đủ nguyên nhân nội bộ.

## Sai công cụ, tham số và hành động thừa

Gold tổng 705 calls; model dispatch 862 calls (nhiều hơn 157, khoảng 22,3%). Có 178 mẫu thiếu ít nhất một tên tool theo gold, 174 mẫu thừa tên tool; hai nhóm có thể giao nhau. Có 51 mẫu lặp cùng call vượt số lần trong gold, là tín hiệu loop cần điều tra, không tự động chứng minh mọi call lặp đều trái yêu cầu.

435 cặp có thể khớp nếu chỉ xét tên tool, nhưng chỉ 190 cặp khớp cả arguments theo exact equality. Tỷ lệ này cho thấy arguments là nút thắt bên cạnh chọn công cụ; text fuzzy hợp lệ có thể khiến phép exact thấp hơn chấm thi.

Ví dụ đối chiếu không lộ giá trị riêng tư:

- `train_0040`: cùng các tool lịch, nhưng lệch `date_from`, `date_to`, `start`, `end` và raw state không khớp.
- `train_0013`: gọi tool thời tiết nhưng khác `location`; raw state vẫn khớp vì thông tin chỉ đọc, minh họa tại sao state không đủ.
- `train_0038`: gold không call, model gọi `smart_home.get_status` hai lần.

22 mẫu gold không call: chỉ 11 mẫu dự đoán đúng danh sách rỗng (50%). Không nên luôn tra cứu để “chắc chắn” nếu yêu cầu ngoài phạm vi.

## Chuỗi tác vụ dài

| Số gold call | Mẫu | Hash khớp |
|---:|---:|---:|
| 0 | 22 | 17 |
| 1 | 77 | 41 |
| 2 | 86 | 31 |
| 3 | 70 | 16 |
| 4 | 28 | 4 |
| 5 | 22 | 0 |
| 6 | 4 | 0 |

Chất lượng giảm mạnh khi chuỗi dài; 26 mẫu cần 5–6 call chưa mẫu nào khớp hash. Cần ưu tiên giữ tiến độ tác vụ qua observations, grounding ID và dừng đúng. Chưa thể quy toàn bộ lỗi này cho context dài hoặc ASR; ASR không nằm trong run.

## Trạng thái dừng và latency

| Stop reason | Số mẫu | Hash khớp |
|---|---:|---:|
| finished | 230 | 80 |
| asked_missing_info | 12 | 2 |
| invalid_action | 30 | 17 |
| deadline | 19 | 3 |
| max_calls | 18 | 7 |

`finished` là quyết định dừng, không phải chứng nhận hoàn thành. Trong 230 mẫu finished, chỉ 80 hash khớp, 26 có toàn bộ call/arguments exact. `invalid_action` nghĩa nhiều lần parse không thành công, nhưng run không lưu raw output nên chưa biết chủ yếu là markdown, JSON sai, action ngoài giao thức hay cutoff.

59 mẫu phát sinh ít nhất một lỗi parse/validation, trong đó một số đã repair được. 19 timeout = 6,1%; 18 mẫu chạm max_calls=8, trong khi gold dev tối đa 6 calls, gợi ý gọi thừa/loop nhưng không phải bằng chứng chỉ cần tăng giới hạn là đúng.

Controller latency: trung bình 11,95 s, p50 9,47 s, p95 29,70 s, max 29,78 s. Tổng thời gian controller khoảng 61,6 phút. Chưa tính model loading, xử lý/evaluation ngoài controller; không dùng số này để khẳng định toàn pipeline luôn dưới 30 s/mẫu. Budget timeout là soft, chưa có watchdog cưỡng bức.

## Nhận xét và thứ tự cải thiện

Baseline đủ làm mốc tái lập, nhưng chưa đủ tốt để đưa thẳng vào audio pipeline. Điểm yếu nằm ở policy hỏi/ngoài phạm vi, chọn và điền tham số, theo dõi chuỗi nhiều bước, JSON và dừng. Chỉ tăng tốc GPU không giải quyết phần lớn sai ngữ nghĩa; chỉ fine-tune khi chưa sửa giao thức/đo lường có thể che lỗi tích hợp.

1. Bổ sung scorer BTC hoặc kiểm chứng fuzzy/one-to-one matcher, giữ riêng hash match, raw-state match và exact diagnostic. Không gọi 9,7% hay 35,3% là điểm thi.
2. Dùng journal từng mẫu đã bổ sung; lưu error category và raw action có kiểm soát trong Outputs, không log công khai thông tin người dùng. Cần quan sát observations để biết vì sao một call `ok=true` nhưng `done=false` bị model hiểu nhầm là thành công.
3. Thử structured JSON/action constraints và repair rõ ràng để giảm 30 mẫu invalid_action. Chỉ ép cú pháp không bảo đảm tool/arguments đúng.
4. Cải thiện prompt/policy hỏi thiếu, không bịa ID, không hành động ngoài phạm vi; chuẩn hóa giải ngày giờ từ context và nguồn slot. Chạy ablation cùng model/seed/manifest.
5. Thêm phát hiện vòng lặp/no-progress và kiểm soát các tác vụ đã hoàn thành qua observation; không tự xóa call đã chạy hoặc tăng max_calls mù quáng.
6. Sau các kiểm tra này mới thử LoRA/QLoRA bằng 1.236 train IDs, không đưa 309 dev IDs vào train. Giữ dev cố định để so chất lượng/latency; ASR thêm sau text baseline.

Không sửa cấu hình hoặc nộp lại trong quá trình run 04. Mọi thay đổi tiếp theo là run mới với prompt/config/version ghi riêng.
