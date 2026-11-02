# Trajectory dataset và completion loss mask

Ngày triển khai: 02/11/2026. Bước này tạo dữ liệu SFT và kiểm tra labels; **chưa huấn luyện hoặc cập nhật trọng số model**. Format cố định `custom_json`, `policy off`, cùng SYSTEM prompt và cấu trúc input của baseline hiện tại.

## 1. Đầu vào và đầu ra

Đầu vào:

- Kit train v3.2.0, catalog và VERSION; các source fingerprints phải khớp manifest.
- Manifest train/dev cố định: 1.236 train, 309 dev và các groups đã kiểm chứng trước đó.
- Local tokenizer Qwen3-4B-Instruct-2507; không load weight shards hoặc gọi API.
- Danh sách train IDs đã dùng chẩn đoán, qua `--seen-ids`, để giữ cả nhóm của chúng trong phần train nội bộ.

```text
Outputs/009-trajectory-dataset/run-20261102-01/
  inner_split.json          # inner train/validation IDs, groups, exclusions
  train.jsonl               # prompt/completion từng quyết định
  validation.jsonl
  train.tokens.jsonl        # input_ids, attention_mask, labels và boundaries
  validation.tokens.jsonl
  quarantine.jsonl          # cả trajectory không phù hợp budget, nếu có
  report.json               # counts, lengths, source/tokenizer provenance
  dataset_manifest.json     # completion marker + file fingerprints
  audit.json                # audit độc lập và CPU toy loss/gradient probe
```

Các file này có transcript/arguments/token IDs nên giữ ngoài Git. Chỉ aggregate đã làm sạch được đưa vào `Docs/Results/trajectory-dataset-v1.json`.

## 2. Tách dữ liệu

Giữ nguyên 309 dev, không tạo SFT examples từ các IDs này. Phần 1.236 train được chia theo groups thành inner train/validation, seed 43, target validation 10%. Không dùng gold tool/action để tạo groups hoặc tối ưu phân phối split.

Nhóm chứa một ID trong `--seen-ids` được giữ toàn bộ ở train nội bộ. Với 16 diagnostic IDs của bước trước, có 17 IDs trong các nhóm liên quan được giữ ở train. Split tạo ra **1.112 train và 124 validation**.

Validation này được khóa mới cho giai đoạn SFT, nhưng là **retrospective validation**: train đã được inspect/audit trước đó. Không gọi nó là dữ liệu hoàn toàn chưa từng quan sát. Grouping kế thừa heuristics cũ, chưa chứng minh hết trùng ngữ nghĩa/speaker; không có bảo đảm stratification.

Data Loader có thêm `iter_training(sample_ids=...)`. Nó lọc ID trước khi dựng/kiểm tra `Reference` của mẫu bị loại. Do train/dev nằm chung một JSONL, reader vẫn stream/parse metadata của file; không export hoặc dùng reference của 309 dev để tạo training examples.

## 3. Một mẫu tạo ra những examples nào?

Ví dụ gold có ba calls `A → B → C`:

| Ví dụ | Input hiện tại | Target |
|---|---|---|
| 0 | request/context/history/tools | call A |
| 1 | input + action/result A | call B |
| 2 | input + action/result A, B | call C |
| 3 | input + action/result A, B, C | finish |

Không thêm kết quả B/C vào ví dụ 0 hoặc 1. Không đưa `label`, `final_state` hay reference object vào model input. History đã phản ánh trạng thái ban đầu, chỉ serialize làm ngữ cảnh, không replay.

Hai trường hợp đặc biệt:

- Gold calls rỗng: một example, target `{"type":"finish"}`.
- Gold kết thúc bằng `sys.ask_missing_info` thành công: target cuối là ask; **không thêm finish** vì controller dừng ngay sau ask. Ask không nằm cuối hoặc bị reject sẽ báo lỗi thay vì cắt chuỗi âm thầm.

Raw row có `prompt` dạng hai messages system/user, `completion` dạng một assistant message chứa action JSON, kèm sample ID/step/target metadata. Payload và events được tạo bằng đúng hàm input/prompt của inference, với các gold prefix observations trong phạm vi training được phép.

Đây là teacher-forced data: prefix đúng do BTC cung cấp, không chứng minh model xử lý được lỗi do chính nó tạo ra. Các quyết định là rows độc lập, chưa weighted theo sample nên trajectory dài đóng góp nhiều rows hơn.

## 4. Tokenization và labels

Builder render hai chuỗi bằng template local:

1. Prompt có assistant generation header, giống inference.
2. Prompt + assistant action, không thêm generation header mới.

Hai prefix phải khớp cả text và token IDs. Tokenizer mặc định phải cho prefix giống cách encode training; không chấp nhận boundary BPE lệch rồi đánh dấu labels theo một độ dài ước lượng.

```text
input_ids: [ SYSTEM / USER / HISTORY / EVENTS / ASSISTANT_HEADER ][ ACTION_JSON ][ EOS ]
labels:    [                       -100                         ][ SAME_IDS    ][ EOS ]
```

- Chỉ action JSON và EOS có loss; SYSTEM, user, history, schema, observations và assistant header đều là `-100`.
- Không supervise newline ngoài EOS do template thêm sau message.
- Decode completion phải bằng đúng action JSON; EOS thật phải nằm trong target, không bị mask.
- Raw chat control delimiters trong data bị từ chối, không escape riêng khi training gây khác inference.
- `collate_actions` right-pad input IDs, attention=0 và labels=-100 cho padding. Nếu pad ID bằng EOS ID thì EOS thật vẫn supervised; không mask theo giá trị token ID.
- Metadata như sample ID/step không đi vào model inputs. Collator chỉ trả `input_ids`, `attention_mask`, `labels`; tùy chọn `return_tensors='pt'` tạo CPU long tensors.

Causal LM dùng logits tại j-1 dự đoán label j. Vì vậy logit ở cuối prompt vẫn nhận gradient để dự đoán **token đầu tiên của action**, dù labels của các prompt tokens đã bị mask. Không tự shift labels lần nữa trong collator/trainer nếu model đã shift loss nội bộ.

Hướng dữ liệu prompt/completion và completion-only loss phù hợp với [TRL SFT](https://huggingface.co/docs/trl/sft_trainer); serialization dựa [Transformers 4.56.2 chat templates](https://huggingface.co/docs/transformers/v4.56.2/en/chat_templating). Phiên bản trainer về sau phải được kiểm tra để không tokenize/mask/truncate lại các pretokenized labels này.

## 5. Budget và quarantine

Mỗi row phải thỏa:

- Prompt tokens + 384 ≤ 16.384, theo reserve của LocalPlanner.
- Action JSON + EOS ≤ 384 tokens.
- Toàn sequence ≤ `--max-length`, mặc định 16.384.
- Trajectory nằm trong 8 tool calls/12 decisions của controller.

Nếu vượt budget, mặc định quarantine **cả trajectory**, ghi lý do, step đầu vi phạm và token lengths. Không export một nửa trajectory hoặc chuyển mẫu validation sang train để tận dụng. Reserved split vẫn giữ nguyên, report phân biệt retained/quarantined coverage.

`--overflow-policy error` dừng build. Các file partial có thể còn lại để điều tra nhưng **không có dataset_manifest complete**, không được dùng training. Output directory phải mới; không ghi đè/reuse partial run.

Các lỗi template, prefix hoặc dữ liệu không được coi là overflow để âm thầm bỏ qua. Không truncate schema/history/target. Việc export được sequences dài không chứng minh QLoRA batch đó vừa 16 GB VRAM; memory smoke thuộc bước trainer tiếp theo.

## 6. Lệnh tạo và audit

PowerShell từ repo:

```powershell
Set-Location D:\Projects\lifemate-workspace\lifemate-agent
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"

.venv/Scripts/python.exe -m lifemate.cli prepare-trajectories `
  --data-root ../Data `
  --manifest ../Outputs/004-train-dev-split/run-20261102-01/split_manifest.json `
  --model-path ../Models/Qwen3-4B-Instruct-2507 `
  --seen-ids ../Outputs/007-qwen-native-protocol/run-20261102-02-paired-train16/ids.json `
  --validation-fraction 0.1 --seed 43 `
  --output-dir ../Outputs/009-trajectory-dataset/my-new-dataset

.venv/Scripts/python.exe -m lifemate.cli audit-trajectories `
  --dataset-dir ../Outputs/009-trajectory-dataset/my-new-dataset `
  --model-path ../Models/Qwen3-4B-Instruct-2507 `
  --torch-probe `
  --output ../Outputs/009-trajectory-dataset/my-new-dataset/audit.json
```

`--seen-ids` có thể lặp lại để hợp nhất nhiều danh sách; IDs phải thuộc outer train. `--max-length` thấp hơn được phép và sẽ quarantine các trajectory dài theo giới hạn đó. Chưa có lệnh train trong bước này.

## 7. Audit kiểm tra điều gì?

Audit độc lập đọc raw/tokens và kiểm tra fingerprints, đúng partition/group, không có outer dev hoặc seen IDs trong validation, rows đầy đủ/đúng thứ tự, terminal targets và prefix nối tiếp đúng action trước đó.

Nó tokenize lại bằng tokenizer thật, đối chiếu prompt IDs, full sequence, target decode, labels, EOS, attention và budgets; phát hiện prompt label bị unmask hoặc event tương lai ngay cả trong negative tests đã cập nhật lại file hash.

`--torch-probe` dùng logits toy có vocabulary 7 trên CPU với labels đã map chỉ để thử causal loss/gradient. Positions có label -100 phải có loss/gradient bằng 0; positions supervised phải có gradient. Đây **không phải LLM loss**, không tải model/optimizer và không cập nhật weights.

Gradient ở đây là gradient đối với toy logits tại các vị trí dự đoán sau causal shift. Prompt vẫn tham gia forward và có thể nhận gradient gián tiếp qua attention trong LLM thật; loss mask không loại prompt khỏi computation graph.

Audit là kiểm tra nhất quán và provenance, không chữ ký bảo mật hay replay chứng minh mọi observation nguồn đều đúng. Observations gold đã được toolkit kiểm chứng ở các bước trước; audit này không mở ToolEnv hoặc đọc app state. Training loop tương lai cần kiểm tra lại labels có thật sự được sử dụng nguyên vẹn.

## 8. Kết quả thực tế

Run: `Outputs/009-trajectory-dataset/run-20261102-01/`.

| Chỉ số | Train nội bộ | Validation nội bộ |
|---|---:|---:|
| Samples | 1.112 | 124 |
| Decision examples | 3.517 | 392 |
| Call targets | 2.551 | 285 |
| Finish targets | 966 | 107 |
| Samples cần ask | 146 | 17 |
| Samples ngoài phạm vi | 82 | 7 |
| Max prompt tokens | 10.092 | 9.297 |
| Max target tokens, gồm EOS | 105 | 105 |
| Max sequence tokens | 10.099 | 9.304 |

Tổng **3.909 rows**, 108.117 supervised tokens và 22.278.440 ignored prompt tokens. Không quarantine mẫu nào; 309 outer dev IDs không xuất hiện trong SFT datasets. Có 16 seen IDs và 1 companion trong cùng nhóm được giữ ở train.

Audit toàn bộ 3.909 rows đạt. CPU toy loss/gradient probe đạt, gồm 167 supervised shifted positions và 29.461 ignored shifted positions trong batch kiểm tra. Giá trị toy loss khoảng 1,94591 không biểu thị chất lượng Qwen.

## 9. Mã nguồn và kiểm thử

| Module | Vai trò |
|---|---|
| `training/trajectories.py` | Inner split và per-decision prompt/completion |
| `training/loss_mask.py` | Encode, mask validation, collator và CPU probe |
| `training/prepare_trajectories.py` | CLI build, budget/quarantine, provenance và fingerprints |
| `training/audit_trajectories.py` | CLI audit artifact độc lập |
| `src/lifemate/input/adapter.py` | Selected-ID reference iteration |

Prototype qua 7 tests trước tích hợp. Suite sau tích hợp: **102 tests đạt** khi bật toolkit/tokenizer contract tests; negative cases gồm tương lai lọt prefix, unmask prompt, EOS=pad, boundary lệch, budget, group leakage và held-out Reference cố ý hỏng.

Wheel đã build và kiểm tra các module mới khớp source; hai CLI help chạy ngoài cây mã nguồn. Bằng chứng tests/build ở `Outputs/009-trajectory-dataset/run-20261102-02-verification/`. Không tải thêm thư viện huấn luyện trong bước này.

```powershell
$env:LIFEMATE_DATA_ROOT = 'D:\Projects\lifemate-workspace\Data'
$env:LIFEMATE_MODEL_PATH = 'D:\Projects\lifemate-workspace\Models\Qwen3-4B-Instruct-2507'
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Bước tiếp theo: trainer QLoRA với memory smoke, kiểm tra loss trên vài batches, save/reload adapter và đánh giá closed-loop. Phải dùng inner validation để chọn checkpoint và giữ 309 dev cho các mốc; chưa có kết quả fine-tune hoặc bằng chứng tăng điểm trong bước chuẩn bị dữ liệu này.
