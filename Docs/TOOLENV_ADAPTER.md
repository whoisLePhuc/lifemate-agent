# ToolEnv Adapter

> Các đường dẫn `../Outputs/` và `../Experiments/` ghi trong nội dung dưới đây được tính từ repo `lifemate-agent`, không từ thư mục Docs. Chúng là bằng chứng workspace ngoài Git; lệnh triển khai nằm trong repo này.

Adapter nối controller với `zaic_tools` chính thức. Không có dependency ngoài thư viện chuẩn Python; đã chạy kiểm tra bằng Python 3.13.12. Mã dùng cú pháp Python 3.10+ nhưng chưa chạy trên mọi phiên bản.

## Chức năng

- Chọn runtime đúng thư mục `Data/<split>/zaic_tools`; chặn việc trộn train/public runtime trong cùng tiến trình Python.
- Tạo đúng một ToolEnv cho mỗi bộ `(data_root, split, sample_id)` trong tiến trình; chặn tạo lại kể cả sau close hoặc khởi tạo lỗi.
- `validate()` gọi validator BTC, không thực thi và không ghi call.
- `call()` dispatch một lần tới ToolEnv; trả nguyên payload `ok`, `done`, dữ liệu/lỗi từ runtime qua bản sao.
- `get_tool_calls()` lấy sổ lời gọi từ môi trường, bao gồm call bị từ chối.
- `finalize()` lấy call và hash từ cùng môi trường, trả record submission rồi chặn hành động mới.
- Khi runtime raise exception bất ngờ, khóa session ở `broken`, không retry hoặc tạo hash giả.

Không truy cập trực tiếp file trạng thái, không expose `get_state()`, reset hoặc replay history. Runtime BTC tự nạp state và cập nhật trong bộ nhớ theo hợp đồng của kit.

## Dùng với Data Loader

Từ repository này, PowerShell:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
```

```python
from lifemate_data import DatasetLoader
from lifemate_tools import ToolEnvAdapter

data_root = r"D:\Projects\lifemate-workspace\Data"
split = "train"
loader = DatasetLoader(data_root, split)

for item in loader.iter_inputs():
    with ToolEnvAdapter(data_root, split, item.id) as env:
        # ASR/controller chọn name và arguments từ item.tools.
        # validation = env.validate(name, arguments)
        # if validation['ok']:
        #     observation = env.call(name, arguments)
        # Controller đọc observation, rồi tiếp tục hoặc dừng.
        record = env.finalize()
    # record gồm id, tool_calls, state_hash; writer có thể ghi ra JSONL.
```

Đoạn trên chưa có agent nên tạo kết quả không hành động, không phải lời giải tác vụ. Không gọi trực tiếp `ToolEnv` ở nơi khác khi đã dùng adapter.

## Giao diện và vòng đời

| API | Đầu vào | Đầu ra |
|---|---|---|
| `ToolEnvAdapter(data_root, split, sample_id)` | Root chứa các split, tên split và ID | Session active |
| `validate(name, arguments)` | string và JSON object | dict của validator BTC |
| `call(name, arguments)` | string và JSON object | observation dict của tool BTC |
| `get_tool_calls()` | Không | list call đã dispatch |
| `finalize()` | Không | dict `{id, tool_calls, state_hash}` |
| `close()` | Không | Giải phóng handle; không reset môi trường |
| `status` | Không | `active`, `finalized`, `broken` hoặc `closed` sau khởi tạo |

`finalize()` được cache: gọi lại trả bản sao cùng record, không đọc hash lần nữa. Record đã finalize vẫn đọc lại được sau close. Nếu close trước finalize thì không lấy được record; context manager không tự finalize khi thoát, nhằm tránh che lỗi đang xảy ra.

`arguments` phải là object JSON với key string và số hữu hạn; lỗi Python/JSON được chặn trước dispatch và không ghi call. **Không tự validate schema trong `call()`**: nếu controller gọi thẳng một tool sai schema, runtime từ chối và ghi nó vào ledger theo thể lệ. Controller muốn tránh call sai phải dùng `validate()` trước khi dispatch.

`ok=True` không đồng nghĩa tác vụ hoàn thành: runtime có thể trả `done=False` hoặc danh sách rỗng. Adapter không biến kết quả nghiệp vụ thành lỗi hạ tầng.

## Lỗi và giới hạn

- `AdapterError`: cấu hình, tên tool hoặc arguments không đúng contract adapter.
- `LifecycleError`: thao tác sau finalize/close/broken, tạo lại mẫu hoặc runtime sai split.
- `RuntimeFailure`: khởi tạo/thực thi/hash của runtime bất ngờ lỗi hoặc sai định dạng. Không tự retry; caller đánh dấu mẫu lỗi. Nếu call có thể đã có side effect, không xuất một record thành công suy đoán.
- Guard một-mẫu-một-instance chỉ có hiệu lực trong một tiến trình. Runner phải chia ID không trùng giữa worker và không reset/retry mẫu trong cùng lần inference bằng cách mở tiến trình mới.
- Train và public test chạy ở tiến trình riêng. Tham số `toolenv_factory` chỉ là điểm tiêm fake runtime cho unit test; production bỏ trống.
- Adapter chưa có timeout, watchdog, controller, validation ngữ nghĩa hoặc JSONL writer. `RLock` chỉ tuần tự hóa thao tác trong một session, không thay cơ chế deadline.
- Private split được hỗ trợ về tên nhưng chưa có dữ liệu để xác minh.
- Không import runtime từ nguồn không tin cậy: đó là mã Python do BTC cấp, không phải dữ liệu JSON thuần.

## Kiểm tra chạy lại

Chọn output mới ở mỗi lần chạy:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v

# Opt-in kiểm tra runtime thật trên một mẫu train, trong tiến trình mới.
$env:LIFEMATE_DATA_ROOT = 'D:\Projects\lifemate-workspace\Data'
python -m unittest discover -s tests -p test_runtime_contract.py -v
Remove-Item Env:LIFEMATE_DATA_ROOT

# Replay train trong tiến trình riêng; không phải agent chọn action.
python scripts/check_adapter.py --data-root 'D:\Projects\lifemate-workspace\Data' --split train --output 'D:\Projects\lifemate-workspace\Outputs\002-toolenv-adapter\run-new\train-report.json'

# Smoke public không gọi action, không dùng/tạo nhãn test.
python scripts/check_adapter.py --data-root 'D:\Projects\lifemate-workspace\Data' --split public_test --limit 3 --output 'D:\Projects\lifemate-workspace\Outputs\002-toolenv-adapter\run-new\public-smoke.json'
```

## Bằng chứng đã có

Thử nghiệm nguồn: `../Experiments/002-toolenv-adapter/` trong workspace; kit `v3.2.0`, catalog `2.7.0`.

- 9 unit tests adapter đạt; 10 unit tests Data Loader hiện có vẫn cần duy trì.
- 1 contract test runtime thật đạt: validate không ghi call; dispatch sai schema có trong submission; finalize chặn call mới.
- Replay 1.545 mẫu train, 3.541 call: ledger khớp nhãn 1.545/1.545; kết quả tool khớp chính xác 1.545/1.545; không có gold call bị validation từ chối.
- Mỗi mẫu nhận hash SHA-256 đúng định dạng; kiểm tra này không so hash với gold.
- Smoke 3 mẫu public: mở môi trường, lấy ledger rỗng/hash, finalize hợp lệ; không sinh file dự đoán hoặc đánh giá chất lượng.
- Chặn tạo lại cùng sample và trộn runtime split đã được kiểm tra.
- Báo cáo thử nghiệm: `../Outputs/002-toolenv-adapter/run-20261102-01/`.
- Sau tích hợp, 20/20 tests (10 loader, 9 adapter, 1 runtime thật) đạt; replay toàn train và smoke public vẫn đạt. Bằng chứng: `../Outputs/002-toolenv-adapter/run-20261102-02-integrated/{tests.txt,train-report.json,public-smoke.json}`.

Các checks adapter ở trên không đối chiếu toàn bộ final state hoặc đo model latency. Phần đó đã được kiểm tra riêng trong [replay train](REPLAY_SUBMISSION.md) và [baseline text](TEXT_BASELINE.md); chưa tính điểm thi chính thức. Khi chạy unittest, kit có thể phát `ResourceWarning` do cách BTC mở file trong constructor; không sửa toolkit để che cảnh báo này.
