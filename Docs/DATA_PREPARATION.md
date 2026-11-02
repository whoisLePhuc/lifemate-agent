# Chia train/dev cố định và kiểm tra audio

## Chạy lại

Từ repo chính, Python 3.10+ (đã kiểm tra bằng 3.13.12), không dependency ngoài:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
python scripts/prepare_data.py --data-root 'D:\Projects\lifemate-workspace\Data' --output-dir 'D:\Projects\lifemate-workspace\Outputs\004-train-dev-split\run-new' --seed 42 --dev-fraction 0.2
```

Thư mục output phải mới và ngoài Data. Không sửa dữ liệu BTC, không import ToolEnv, không đọc môi trường. Public test chỉ được kiểm tra audio, không dùng để chọn split hoặc tạo nhãn.

## Sản phẩm

- `audio_audit.json`: số file được tham chiếu, định dạng, thời lượng và lỗi decode/frame. Đọc hết WAV theo chunk, đối chiếu số byte với frame count; không chỉ xem header.
- `split_manifest.json`: danh sách `train_ids`, `dev_ids`, các nhóm, seed, thuật toán `grouped-v1` và SHA-256 của VERSION/train JSONL/catalog.
- `split_report.json`: quy mô nhóm, bằng chứng trùng, phân bố history, số call, hỏi thiếu, ngoài phạm vi theo gold và các app. Chỉ label train được dùng cho thống kê phân bố.

Manifest và báo cáo nằm trong Outputs, không commit raw data hoặc nhãn. Đóng băng manifest hiện tại cho các thí nghiệm so sánh; nếu nguồn thay đổi, so fingerprint và tạo phiên bản split mới, không âm thầm dùng lại hoặc ghi đè. Fingerprint metadata không chứng minh audio chưa bị thay đổi; chạy lại audit khi nhận gói mới.

## Luật nhóm

Nối mẫu thành connected components nếu có một trong các bằng chứng:

1. Transcript lượt cuối giống nhau sau NFC/lowercase/tách token, hoặc toàn chuỗi lời user giống nhau.
2. Có lời user trùng từ 6 token trở lên ở lượt cuối/history; nối bắc cầu để các nhánh cùng lịch sử không bị tách.
3. Nội dung PCM của audio lượt cuối giống nhau, kèm channels/rate/sample width.
4. Transcript lượt cuối từ 8 token trở lên, Jaccard tập token trigram >= 0,90.

Giữ số và tên riêng, không thay bằng placeholder. Không dùng gold calls/results/state để tạo nhóm. Không dùng một câu history ngắn như “đồng ý” để nối mọi hội thoại. Dù vậy việc nối theo câu dài có thể gom các tình huống độc lập có câu nói phổ biến; đây là cách bảo thủ để giảm rò rỉ.

Sắp xếp nhóm theo ID rồi shuffle bằng `random.Random(seed)`, chọn nguyên nhóm khi làm số dev gần mục tiêu hơn. Không chia lẻ nhóm để đủ tỷ lệ. Chương trình kiểm tra disjoint/coverage, nhóm không cắt qua hai phía và tái tạo cùng split khi đảo thứ tự input.

Đây là heuristics, chưa có conversation ID chính thức được xác minh. Không bảo đảm loại hết paraphrase, cùng người nói hoặc cùng ngữ cảnh nghiệp vụ. Audio trùng ở history chưa được dùng làm khóa nối độc lập; history được nhóm bằng transcript. Trigram là tương đồng bề mặt, không phải semantic deduplication. Không stratify theo nhãn; báo phân bố để thấy lệch, không dò nhiều seed để chọn theo điểm model.

## Sử dụng split cho baseline

```python
import json
from lifemate_data import DatasetLoader

with open(manifest_path, encoding='utf-8') as stream:
    manifest = json.load(stream)
dev_ids = set(manifest['dev_ids'])
for item in DatasetLoader(data_root, 'train').iter_inputs():
    if item.id in dev_ids:
        run_agent(item)
```

Huấn luyện chỉ dùng `train_ids`. Reference dev chỉ vào evaluator; không truyền gold action/result/state cho agent. Baseline transcript sạch sẽ có đường cung cấp transcript chuẩn riêng, chưa được triển khai trong bước này.

## Kết quả đã kiểm chứng

Gói BTC v3.2.0, catalog 2.7.0; báo cáo tại `Outputs/004-train-dev-split/run-20261102-01/` tính từ workspace root:

| Chỉ tiêu | Kết quả |
|---|---:|
| Train nội bộ | 1.236 mẫu |
| Dev | 309 mẫu (20%) |
| Tổng nhóm | 1.490 |
| Nhóm có nhiều mẫu | 38 |
| Nhóm lớn nhất | 8 mẫu |
| Nhóm cắt qua train/dev | 0 |
| Dev có history | 201 |
| Dev gold rỗng | 22 |
| Dev có hỏi thiếu thông tin | 54 |

Đọc đủ audio được tham chiếu ở cả hai split, không có lỗi đọc/frame. Điều này chưa đánh giá tiếng nói, im lặng, clipping hay chất lượng ASR. Không có audio lượt cuối trùng PCM được phát hiện trong train. Các loại bằng chứng trùng có thể đếm cùng cặp nhiều lần, không cộng chúng để suy ra số mẫu trùng.

34/34 tests tích hợp đạt (bao gồm 1 test runtime opt-in khi đặt LIFEMATE_DATA_ROOT); log `Outputs/004-train-dev-split/verification-20261102/tests.txt`. Sáu tests mới kiểm tra Unicode/trùng text, quan hệ history bắc cầu, audio/near-text, split disjoint và tái lập, thứ tự input, WAV bị cắt cụt. Nếu không bật biến môi trường cho test runtime, test đó được skip.

Mã thử nghiệm và lệnh gốc: `Experiments/004-train-dev-split/README.md` trong workspace. Mã preparation đã tích hợp khớp snapshot đã chạy; chưa có LLM baseline hoặc điểm dev.
