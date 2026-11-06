# Huấn luyện lại adapter theo định dạng tool-calling native của Qwen3.5

Branch: `experiment/native-retrain`. Tài liệu này là kế hoạch thực hiện trên một máy GPU khác. Branch đã có sẵn đường inference native và đã được kiểm chứng. **Phần dựng dữ liệu huấn luyện native chưa được viết**: mục 4 mô tả chính xác những chỗ cần sửa.

## 1. Bằng chứng và giả thuyết

Thí nghiệm 031 của workspace chạy trên 124 mẫu inner-validation, dùng transcript gold (không ASR), hạn 30 s mỗi mẫu, thinking tắt.

| Nhánh | Model | Giao thức | Điểm local | Khớp hash |
|---|---|---|---|---|
| A | base Qwen3.5-4B | `custom_json` | 0,2299 | 51 |
| C | base Qwen3.5-4B | `qwen35_native` | 0,3922 | 72 |
| G (production) | adapter 0880 | `custom_json` | **0,7344** | **101** |
| J | adapter 0880 | `qwen35_native` | 0,5581 | 80 |

- Chỉ đổi sang native, base tăng +0,162 (CI 95% [0,096; 0,230], p < 0,001). Phần tăng nằm ở mẫu `single` (0,40 → 0,64) và `multi` (0,17 → 0,36); mẫu `ask` gần như không đổi (khoảng 0,10).
- Adapter 0880 huấn luyện trên `custom_json`. Ép nó sang native thì giảm 0,176, vì nó vẫn xuất JSON.
- Mức tăng của adapter so với base (+0,505) đến chủ yếu từ dữ liệu huấn luyện. Ví dụ hỏi lại: `ask` từ 0,10 lên 0,66.

**Giả thuyết.** Một adapter học cùng dữ liệu đó nhưng ở định dạng native sẽ giữ được kiến thức nghiệp vụ (hỏi lại, chọn tham số) và cộng thêm phần lợi của định dạng mà base đã quen.

**Kết quả có thể xảy ra.** Mức tăng của native có thể chỉ có ích cho base chưa huấn luyện, và biến mất khi đã fine-tune. Kết quả hòa vẫn là một kết quả hợp lệ.

## 2. Những gì đã có trên branch

| Thành phần | Vị trí |
|---|---|
| Giao thức `qwen35_native`: tools truyền qua chat template; lịch sử gửi `arguments` dạng dict; prompt hệ thống `NATIVE_SYSTEM` (vi) và `NATIVE_SYSTEM_EN` | `src/lifemate/agent/protocol.py` |
| Parser XML `parse_native_xml`: đúng một `<tool_call>`, ép kiểu theo schema, từ chối tham số trùng; văn bản thường = finish | `protocol.py` |
| Controller nhận `protocol='qwen35_native'` và `lang` | `src/lifemate/agent/controller.py` |
| `Qwen35Planner(thinking=..., think_budget=...)` (thinking chỉ để thử nghiệm, **giữ tắt**) | `src/lifemate/agent/qwen35_planner.py` |
| `run-agent --protocol qwen35_native --prompt-lang --thinking --partition train --inner-split` | `src/lifemate/pipeline/run_agent.py` |
| 17 unit test | `tests/unit/test_qwen35_native.py` |

Branch **không** chứa H4 (`finish_shadow_v1`).

Unit test: `python -m unittest discover -s tests/unit -t .` chạy 178 test. Trong môi trường không có `peft` sẽ có đúng 4 lỗi import, và chúng có sẵn từ trước.

Audit round-trip (gold → template Qwen3.5 → `parse_native_xml`) khôi phục đúng 3541/3541 lời gọi trong `train.jsonl`.

## 3. Artifact cần chép sang máy mới (không có trong git)

Đặt theo bố cục workspace, tức `lifemate-agent/` nằm cạnh `Data/`, `Models/` và `Outputs/`.

| Artifact | Đường dẫn gốc | Kích thước / sha256 |
|---|---|---|
| Dữ liệu cuộc thi (chỉ đọc, không sửa) | `Data/` | 1,1 GB |
| Model base | `Models/Qwen3.5-4B` | 8,8 GB |
| Outer split manifest | `Outputs/004-train-dev-split/run-20261102-01/split_manifest.json` | `44694abd…2b594` |
| Inner split (1.112 train / 124 validation, chính là split của 0880) | `Outputs/016-qwen35-qlora/dataset-20261103-01/inner_split.json` | `0e405b34…127a7` |
| 124 ID đánh giá | `Outputs/031-native-thinking-prompt-lang/inner124-01/ids.json` | `75fb52e7…626c` |
| Adapter đối chứng 0880 | `Outputs/020-one-epoch-training/epoch-seed42-resumed-20261104-01/checkpoint-0880/adapter` | 124 MB |
| Adapter JSON ở bước 400 (cho cổng giữa chừng) | `Outputs/020-one-epoch-training/epoch-seed42-20261104-01/checkpoint-0400/adapter` | 124 MB |

Sau khi chép, chạy `sha256sum` để đối chiếu.

**Môi trường.** Python 3.12, cài theo `requirements-qwen35-training-linux-py312.lock.txt` (Torch 2.8.0+cu128, Transformers 5.22.1, PEFT 0.21.2, bitsandbytes 0.50.2). Cần GPU BF16, trên 16 GiB VRAM (đỉnh 15,5 GiB ở run 0880) và ít nhất 20 GiB đĩa trống.

**Cách dùng gói.** Theo AGENTS.md, cài `lifemate-agent` thành package (`pip install --target <dir> .`) và dùng gói đã cài, không trỏ `sys.path` vào thư mục nguồn.

## 4. Việc cần lập trình

Nguyên tắc: **token lúc huấn luyện phải trùng từng token với prefix lúc inference**. Mọi prompt đều dựng bằng đúng `build_messages(payload, events, 'qwen35_native')` và `tool_schemas(payload)` mà controller dùng, rồi render qua `tokenizer.apply_chat_template(..., tools=..., enable_thinking=False)`.

Thêm `--protocol {custom_json,qwen35_native}` (mặc định `custom_json`) cho `prepare-trajectories`, `audit-trajectories` và `train-epoch`. Đường `custom_json` phải giữ nguyên byte. Kiểm bằng cách dựng lại dataset 0880 và so với `selection.json`, như thí nghiệm 024 đã làm.

### 4.1 `training/trajectories.py` · `trajectory_rows`

- **Giữ nguyên:** trình tự hành động (các gold call, rồi `finish` trừ khi call cuối là `sys.ask_missing_info`), mọi kiểm tra hiện có, và cách tách prefix.
- **Prompt:** `prompt = build_messages(payload, events, 'qwen35_native')`, và lưu thêm `tools = tool_schemas(payload)` vào hàng.
- **Đích cho một call:** message assistant có `tool_calls=[{'type':'function','function':{'name':..., 'arguments': <dict>}}]` và `content=''`. Template tự render XML.
- **Đích cho finish:** cố định `{'role':'assistant','content':'Đã xong.'}`.
  - Quyết định này cần ghi vào report.
  - Parser coi mọi văn bản thường là finish. Một chuỗi ngắn cố định giúp đích rẻ và ổn định. Không dùng lời đáp tự do, vì dữ liệu không có nhãn cho nó.
- **Kiểm tra cú pháp:** thay `parse_action` (JSON) bằng `parse_native_xml(target_text, tools_by_name)`, và yêu cầu kết quả bằng đúng hành động gold.

### 4.2 `training/loss_mask.py` · `encode_action`

- **Render với tools:** `prefix = apply_chat_template(prompt, tools=tools, add_generation_prompt=True)` và `full = apply_chat_template(prompt + completion, tools=tools)`.
- **Văn bản đích** là `full[len(prefix):]` tới trước `<|im_end|>`. Không còn là `completion[0]['content']`, vì message call có `content` rỗng.
- **Giữ các kiểm tra sau** (chúng là cổng chất lượng):
  - `full` bắt đầu bằng `prefix`;
  - sau đích chỉ còn EOS cộng khoảng trắng;
  - decode các token đích ra đúng văn bản đích;
  - `parse_native_xml(văn bản đích)` bằng hành động gold;
  - ngân sách: `prompt + 384 ≤ 16384` và đích ≤ 384 token.
- **Lưu ý:** inference decode với `skip_special_tokens=False` khi có tools, nên `<tool_call>` phải nằm trong phần được giám sát.

### 4.3 `training/audit_trajectories.py`

Áp dụng cùng thay đổi: render với tools và kiểm bằng `parse_native_xml`.

**Bổ sung một kiểm tra chéo:** với 20 hàng ngẫu nhiên, gọi đúng hàm render trong `Qwen35Planner` (gói đã cài, `enable_thinking=False`, kèm tools). Chuỗi đó phải trùng byte với `prefix` của hàng.

### 4.4 `training/prepare_trajectories.py` và `training/train_epoch.py`

- Ghi `protocol` vào `report.json` và `dataset_manifest.json`.
- `prompt_sha256` dùng `system_prompt(protocol)`, tức `NATIVE_SYSTEM` cho native.
- `train_epoch` hiện so `prompt_sha256` với `SYSTEM` của custom_json. Cần so với prompt của protocol ghi trong manifest, và đưa `protocol` vào config fingerprint.
- `sparse_loss`, `epoch_schedule` và `qwen35.py` không cần đổi.

### 4.5 Test cần thêm

- Hàng native có prefix trùng render của planner.
- Đích call parse ra đúng gold; đích finish là `Đã xong.`.
- Mặt nạ loss chỉ phủ đích cộng EOS.
- Đường `custom_json` không đổi byte so với trước.
- `train_epoch` từ chối dataset có protocol khác với prompt.

## 5. Các cổng kiểm tra trước khi huấn luyện dài (CPU, chạy nhanh)

| Cổng | Điều kiện đạt |
|---|---|
| G1 Dựng dataset native | 1.112 / 124 sample. Số hàng phải bằng 3.517 / 392 như custom_json, vì trình tự hành động không đổi. Quarantine = 0; nếu có hàng bị loại vì ngân sách token thì dừng lại để xem xét. |
| G2 Audit | Mọi hàng: prefix khớp, decode khớp, parse bằng gold. Kiểm tra chéo với planner trùng byte. `--torch-probe` đạt. |
| G3 Độ dài | Ghi phân bố token prompt/đích của native so với custom_json (max 9.905 ở 0880). Thời gian một bước tỉ lệ gần tuyến tính với độ dài, dùng số này để ước tính thời gian. |
| G4 Regression | Đường `custom_json` cho ra `selection.json` trùng run 0880 (cách làm như thí nghiệm 024, tầng 2). |
| G5 Smoke resume | `lifemate verify-resume` trên dataset native: `RESTORE_AND_CONTINUATION_VERIFIED`. |

## 6. Huấn luyện

Dùng cấu hình giống 0880 để chỉ còn **định dạng** là biến thay đổi:
- QLoRA NF4, rank 16 / alpha 32 / dropout 0,05, all-linear;
- AdamW lr 1e-4, weight decay 0, clip 1;
- warmup 44 bước rồi cosine về 10%;
- accumulation 4, tức 880 bước cho 3.517 hàng;
- seed 42, thứ tự một lượt với hàng dài nhất đầu tiên;
- loss chỉ trên đích cộng EOS.

```bash
python -m lifemate.cli train-epoch --protocol qwen35_native --dataset-dir <dataset-native> \
    --model-path ../Models/Qwen3.5-4B --output-dir <run> --preflight
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m lifemate.cli train-epoch --protocol qwen35_native \
    --dataset-dir <dataset-native> --model-path ../Models/Qwen3.5-4B --output-dir <run> --seed 42 \
    --save-every 100 --validation-every 400 --stop-after 400
```

Thời gian: run 0880 mất khoảng 30 s mỗi bước trên GPU cũ, tức khoảng 7,3 giờ cho 880 bước. Ước tính lại theo G3 và GPU của máy mới.

Loss teacher-forced của native **không so được** với loss của custom_json, vì hai bên có tập token đích khác nhau. Chỉ dùng loss để phát hiện lỗi (không giảm, ra NaN), không dùng để chọn checkpoint.

### Cổng giữa chừng ở bước 400 (tiết kiệm khoảng một nửa thời gian)

Lịch LR là hàm của số bước trên tổng 880, nên checkpoint 400 của hai run ở cùng điểm lịch học. Đánh giá closed-loop (mục 7) cả hai trên 124 ID:
- native-0400 với `qwen35_native`;
- JSON-0400 với `custom_json` (adapter 0400 ở mục 3; nó chưa từng được chấm).

Quy tắc dừng, ấn định trước:
- **Dừng** nếu native-0400 thấp hơn JSON-0400 và CI 95% của chênh lệch nằm hoàn toàn dưới 0.
- Ngược lại, chạy tiếp tới 880 bằng `--resume <run>/checkpoint-0400` với output-dir mới.

## 7. Đánh giá

Cùng giao thức với 031: oracle, `--seconds 30`, thinking tắt, prompt tiếng Việt, đúng 124 ID ở mục 3.

```bash
python -u -m lifemate.cli run-agent --split train --mode oracle --partition train \
  --inner-split <inner_split.json> --data-root ../Data --model-path ../Models/Qwen3.5-4B \
  --adapter-path <run>/checkpoint-0880/adapter --manifest <split_manifest.json> --ids-file <ids.json> \
  --protocol qwen35_native --prompt-lang vi --thinking off --seconds 30 --output-dir <eval>/N880
python -m lifemate.cli evaluate-run --data-root ../Data --manifest <split_manifest.json> --partition train \
  --allow-subset --submission <eval>/N880/inference/submission.jsonl --output-dir <eval>/N880/scores
```

- **Chạy lại đối chứng G trên máy mới** (adapter 0880, `--protocol custom_json`), không mượn số 0,7344. Hạn 30 s phụ thuộc tốc độ GPU, nên hai bên phải đo trên cùng phần cứng.
- **Mỗi nhánh phải đạt** trace audit hợp lệ và replay 124/124 hash. `Experiments/031-native-thinking-prompt-lang/compare.py` của workspace thực hiện kiểm tra này, cùng bootstrap 10.000 lần (seed 20261105), sign test và bảng theo nhóm `ask`/`single`/`multi`/`out_of_scope`. Chỉ cần sửa `PAIRS`.
- **Báo cáo thêm:** số mẫu chạm deadline, parse error, số quyết định mỗi mẫu, latency trung bình/p95. Ở 031, native cần nhiều lượt hơn.

**Tiêu chí quyết định, ấn định trước:**
- **Thắng:** điểm trung bình native-0880 trừ G (đo trên máy mới) lớn hơn 0, và **cận dưới CI 95% > 0**, và `ask` không giảm quá 0,05. Chỉ khi thắng mới đánh giá một lần trên outer-dev 309 (giao thức `Docs/outer_dev_protocol_v1.md` của workspace) để xác nhận.
- **Hòa hoặc thua:** giữ `custom_json` + adapter 0880. Native không được ưu tiên khi hòa, vì nó tốn thêm lượt quyết định.
- Không dùng 124 mẫu này để chỉnh siêu tham số rồi báo cáo lại trên chính chúng.

## 8. Rủi ro và giới hạn đã biết

- **Template và parser:** giá trị chuỗi chứa `</parameter>` hoặc `</function>` chưa được escape. CRLF chưa được chuẩn hoá. Parser từ chối văn bản sau `</tool_call>` và nhiều lời gọi trong một lượt. Gold hiện tại không gặp các trường hợp này (round-trip 3541/3541), nhưng model mới có thể sinh ra chúng.
- **Native là một gói thay đổi:** cách đặt schema, render lịch sử, prompt hệ thống và phản hồi sửa lỗi cùng đổi. Nếu thắng, vẫn chưa biết yếu tố nào tạo ra mức tăng.
- **Một seed, một lần chạy, 124 mẫu validation hồi cứu.** Khác biệt khoảng 0,02 nằm trong nhiễu (0680 so với 0880: +0,018, p = 0,75).
- **Phụ thuộc phần cứng:** GPU khác cho trọng số khác dù cùng seed (kernel NF4/BF16 không bit-identical giữa các máy).
- **Quy tắc cũ vẫn áp dụng:** không đưa nhãn gold vào input của agent; mỗi mẫu đúng một ToolEnv; không sửa `Data/` hay các thư mục `*_environment/`; output lưu ở `Outputs/`; không commit trọng số hoặc output lớn.
