# QLoRA trainer và adapter inference

Triển khai ngày 02/11/2026. Trainer đã thực sự chạy **4 optimizer steps**, cập nhật adapter, lưu/nạp lại và ghép adapter vào pipeline gọi tool. Đây là **smoke kỹ thuật**, chưa phải huấn luyện toàn tập hoặc bằng chứng tăng điểm agent.

## 1. Thiết kế

- Dùng PyTorch + PEFT + bitsandbytes, giữ nguyên pretokenized `input_ids/attention_mask/labels` của dataset 009. Không tokenize lại, không truncate hoặc dựng lại loss mask.
- Base model local Qwen3-4B-Instruct-2507; NF4 double quantization, compute BF16; LoRA rank 16, alpha 32, dropout 0,05, target all-linear, bias none.
- Micro-batch 1, hỗ trợ gradient accumulation, AdamW chỉ nhận LoRA parameters, gradient clipping 1,0 và non-reentrant gradient checkpointing.
- Attention chỉ dùng efficient/cuDNN SDPA trong trainer đã kiểm chứng; không fallback sang math attention có bộ nhớ bậc hai khi chuỗi dài. Build PyTorch Windows hiện tại không có FlashAttention.
- Tính loss từ các logits cần dự đoán action/EOS. Vẫn forward toàn bộ context và backprop qua attention; không cắt prefix hoặc giả vờ prompt không cần bộ nhớ.
- Seed Python/NumPy/PyTorch/CUDA cố định, source/config/dependencies/data hashes được ghi. Không bảo đảm bit-identical CUDA reductions giữa mọi máy.

Nguồn: [PEFT quantization/QLoRA](https://huggingface.co/docs/peft/developer_guides/quantization), [bitsandbytes 0.48.1 Windows/CUDA installation](https://huggingface.co/docs/bitsandbytes/v0.48.1/en/installation). Các lựa chọn backend và API đã được kiểm tra thêm bằng code cài local và GPU thật.

## 2. Môi trường riêng

Baseline `.venv` được giữ nguyên. `.venv-training` dùng cùng Python 3.12, kế thừa baseline site-packages bằng một file `.pth`; PEFT và bitsandbytes nằm trong environment overlay riêng.

Các phiên bản đã chạy:

| Thư viện | Phiên bản |
|---|---|
| PyTorch | 2.7.1+cu128 |
| Transformers | 4.56.2 |
| Accelerate | 1.10.1 |
| PEFT | 0.17.1 |
| bitsandbytes | 0.48.1 |

```powershell
Set-Location D:\Projects\lifemate-workspace\lifemate-agent
./scripts/setup_training_env.ps1
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"
```

Setup yêu cầu `uv` và baseline đã cài đủ requirements. Nó không nâng cấp baseline packages. File `.pth` chứa absolute path nên sau khi chuyển workspace cần chạy lại setup. `.venv-training` được ignore trong Git. `requirements-training.txt` pin hai packages bổ sung; lock Windows ghi các phiên bản thư viện thực tế đã resolve, không chứa weights hoặc đường dẫn máy.

TRL/datasets chưa được cài hoặc dùng trong trainer này. Dataset đã có labels explicit, vòng train hiện tại dùng chúng trực tiếp.

## 3. Loss được tính thế nào?

Với target ở vị trí j, causal LM dùng logits ở j-1. `training/sparse_loss.py` lấy tập prediction positions có shifted label khác -100, truyền `logits_to_keep` dạng tensor vào Qwen3 rồi tính cross entropy trên đúng các targets đó.

Cách này giảm tensor logits từ toàn sequence × vocabulary xuống các vị trí supervised × vocabulary. Hidden states của toàn context vẫn cần được tính và nhận gradient gián tiếp. EOS vẫn supervised, prompt/padding labels vẫn -100.

Prototype đã đối chiếu trên một Qwen3 nhỏ khởi tạo ngẫu nhiên: sparse loss và full loss, cùng gradients của mọi parameter, khớp trong tolerance. Đây là kiểm tra tính tương đương loss, không phải benchmark chất lượng model.

Trainer v1 chỉ xác nhận backend model_type `qwen3` hỗ trợ API này. Không tự dùng phương pháp selected-logits với một model family khác chưa kiểm thử.

## 4. Dataset, selection và chống leakage

Trainer yêu cầu dataset_manifest complete, file hashes khớp và một mask audit thành công gắn đúng dataset-manifest hash. Token rows được seek bằng byte offsets, kiểm tra labels/EOS và partition, không nạp toàn bộ tokenized corpus vào GPU.

- Chỉ rows trong inner train được cập nhật gradient.
- Inner validation chỉ chạy eval/no_grad; chưa dùng 309 outer dev để training.
- `--train-limit 0` chọn toàn bộ 3.517 fit rows; số dương chọn subset có seed cố định và luôn bao gồm row dài nhất.
- `--validation-limit 0` đánh giá toàn bộ 392 validation rows; mặc định 4 là smoke subset, không phải full validation.
- Row vượt max sequence không được lọc hoặc cắt âm thầm. Dataset dài đã chuẩn bị phải vừa backend; nếu OOM thì run thất bại để điều tra.

Inner validation là split retrospective đã mô tả ở bước 009, không gọi là pristine held-out benchmark. Teacher-forced loss dùng gold prefixes; không thay thế closed-loop tool evaluation.

## 5. Chạy smoke

```powershell
.venv-training/Scripts/python.exe -m lifemate.cli train-sft `
  --dataset-dir ../Outputs/009-trajectory-dataset/run-20261102-01 `
  --model-path ../Models/Qwen3-4B-Instruct-2507 `
  --max-steps 4 --train-limit 4 --validation-limit 4 `
  --output-dir ../Outputs/010-qlora-trainer/my-new-smoke
```

Thư mục output phải mới và ngoài Data/Models/dataset. Chương trình không gọi ToolEnv khi training và không sửa base weights trên đĩa.

```text
my-new-smoke/
  provenance.json, selection.json
  inner_validation_ids.json
  progress.jsonl
  adapter/
    adapter_config.json
    adapter_model.safetensors
    README.md
  report.json           # chỉ có khi train/save/reload/verification hoàn tất
```

Khi run thất bại, có thể có `failure.json` và partial artifacts; không coi chúng là checkpoint đã kiểm chứng. Existing output không bị sửa khi caller thử dùng lại. Adapter checkpoints không chứa optimizer/RNG resume state; exact resume chưa triển khai.

Theo dõi:

```powershell
Get-Content ../Outputs/010-qlora-trainer/my-new-smoke/progress.jsonl -Tail 3
```

Log có train loss, gradient norm, token lengths, optimizer step, thời gian và CUDA peak allocated/reserved. Micro-loss means được trung bình theo accumulation steps, chưa weighting theo task hay số token toàn effective batch. Không diễn giải loss giữa các steps trên các examples khác nhau là một đường cải thiện có kiểm soát.

## 6. Lệnh chuẩn bị cho lượt train dài

Ví dụ sau **chưa được chạy**:

```powershell
.venv-training/Scripts/python.exe -m lifemate.cli train-sft `
  --dataset-dir ../Outputs/009-trajectory-dataset/run-20261102-01 `
  --model-path ../Models/Qwen3-4B-Instruct-2507 `
  --train-limit 0 --validation-limit 0 `
  --max-steps 880 --gradient-accumulation 4 `
  --learning-rate 0.0001 --save-every 100 `
  --output-dir ../Outputs/010-qlora-trainer/my-new-pilot
```

880×4 = 3.520 micro-batches, xấp xỉ một lượt qua 3.517 fit rows, có lặp nhẹ khi hết danh sách. Checkpoints định kỳ là adapter-only. Lịch học v1 dùng learning rate cố định, chưa scheduler/early stopping hoặc chọn best checkpoint; cần đánh giá các lựa chọn này cho lượt dài.

Smoke dài nhất không chứng minh mọi combination/batch shape vừa VRAM. A30, Linux/container, batching lớn và deployment vẫn chưa được kiểm chứng ở bước này.

## 7. Lưu/nạp lại và inference

Trainer so adapter tensor state trước/sau cập nhật, rồi reload base NF4 + cùng kbit dtype preparation và adapter đã lưu. So exact adapter tensors, teacher-forced argmax IDs và loss cùng probe. Reload check không tự cập nhật thêm weights.

Đây là parity **cùng NF4 backend**. BF16 inference có thể cho logits khác vì base quantization và dtype; không hứa bit-identical giữa NF4 và BF16.

Text runner thêm tùy chọn `--adapter-path`. Khi không có cờ này, vẫn dùng baseline cũ. Adapter local safetensors được load với `is_trainable=false`; base BF16 không bị merge/sửa trên đĩa. Provenance ghi adapter file hashes và kiểm tra chúng không thay đổi trong lượt inference.

```powershell
.venv-training/Scripts/python.exe -m lifemate.cli text-baseline `
  --data-root ../Data `
  --manifest ../Outputs/004-train-dev-split/run-20261102-01/split_manifest.json `
  --model-path ../Models/Qwen3-4B-Instruct-2507 `
  --adapter-path ../Outputs/010-qlora-trainer/my-new-smoke/adapter `
  --partition train --ids-file <inner-validation-subset.json> `
  --output-dir ../Outputs/010-qlora-trainer/my-new-inference
```

IDs file là JSON list thuộc 124 inner-validation IDs, không phải fit rows. Trainer xuất danh sách inner validation đầy đủ trong output; smoke 3 mẫu dùng một subset riêng. Raw report ghi physical partition train vì inner validation nằm trong outer train; phạm vi lựa chọn được xác nhận qua IDs/fingerprints.

Submission được kiểm tra bằng `evaluate-run --partition train --allow-subset` và outer manifest như trước, các prediction/gold workers riêng. Một ToolEnv/ID trong mỗi inference/evaluation worker; trainer không truy cập app state.

## 8. Kết quả đã chạy

Training run: `Outputs/010-qlora-trainer/run-20261102-01-smoke/`.

| Chỉ số | Kết quả |
|---|---:|
| Optimizer steps / fit rows được chọn | 4 / 4 |
| Validation rows được chọn | 4/392 |
| Trainable LoRA parameters | 33.030.144 |
| Tổng parameters, gồm adapter chưa merge | 4.055.498.240 |
| Longest selected sequence | 10.099 tokens |
| Peak CUDA reserved trong training steps | khoảng 12,01 GiB |
| Validation token-mean loss, cùng 4 rows | 0,96492 → 0,53713 |
| Adapter tensors thay đổi | 504 |
| Reload adapter state | exact |
| Reload teacher-forced argmax IDs | giống nhau |

Run mất khoảng 69,85 s từ bắt đầu load model đến hoàn tất training/reload verification, không bao gồm mọi CPU preflight. Train step dài nhất 16,59 s ở sequence 10.099; đây không phải tốc độ full-epoch/A30 hoặc inference deadline.

Adapter BF16 closed-loop smoke trên **3 inner-validation samples**: trace audit đạt, replay đúng 3/3 submitted hashes/ledgers, 0 schema rejection và 0 business failure. Tuy nhiên **mean local score = 0**, chỉ 1 hash match với gold, 0 matched calls. Không có paired base run trên 3 IDs này nên không suy ra tăng/giảm do adapter. Đây là bằng chứng tích hợp, chưa phải model tốt hơn.

Bằng chứng inference/evaluation: `Outputs/010-qlora-trainer/run-20261102-02-inner-smoke/`. Raw adapter, labels, IDs, trace và outputs giữ ngoài Git; aggregate nằm trong `Docs/Results/qlora-trainer-v1.json`.

## 9. Giới hạn và bước tiếp theo

Chưa full training trên 3.517 rows, chưa full teacher-forced validation 392 rows, chưa full closed-loop 124 inner-validation samples hoặc 309 dev. Chưa tích hợp ASR, chọn checkpoint theo task score, RL hay exact optimizer resume.

Bước hợp lý tiếp theo là pilot dài có cấu hình/monitoring cố định, sau đó so base và adapter trên cùng inner-validation IDs bằng closed-loop score, hỏi đúng/hỏi thừa, đa bước và latency. Giữ dataset/mask đã audit; chỉ dùng dev ở mốc đánh giá, không chọn checkpoint bằng test labels.

## 10. Kiểm chứng mã nguồn

110 tests đạt, không skip, với runtime ToolEnv và tokenizer contract tests được bật. Có kiểm thử sparse/full causal loss và gradients trên tiny Qwen3, dữ liệu/audit fingerprint, partition/mask, output failure semantics và chặn adapter path dưới environment. Log: `Outputs/010-qlora-trainer/verification-tests.log`. GPU smoke và replay nêu trên là kiểm chứng tích hợp riêng.

Wheel đã được build, đối chiếu bytes của 6 modules và chạy `train-sft --help` từ bản giải nén ngoài source. Bằng chứng: `Outputs/010-qlora-trainer/package-verification/verification.json`.

## 11. Pilot 100 bước và paired evaluation

Pilot 011 đã hoàn tất: 100 optimizer steps × 4 microbatches = 400 rows trong pool 3.517, chưa full epoch. Full 392-row validation loss 0,737335 → 0,108738, save/reload exact, tổng thời gian 4009,31 s. Adapter: `Outputs/011-qlora-pilot/run-20261102-01/adapter`. Aggregate: `Docs/Results/qlora-pilot-v1.json`.

Đã hoàn tất base-first rồi adapter trên cùng 124 inner-validation IDs, BF16/custom_json/policy off/30 s, independent replay/score mỗi variant. Orchestration prototype/cách chạy ở `Experiments/012-adapter-evaluation`; outputs `Outputs/012-adapter-evaluation/run-20261102-01`. Kết quả paired: mean local score 0,18737 → 0,32133; 31 mẫu tăng, 6 giảm, 87 không đổi; ask TP vẫn 0/17. Cả hai trace hợp lệ và replay 124/124 đạt. Aggregate: `Docs/Results/adapter-evaluation-v1.json`. Chưa đánh giá adapter trên outer dev; không coi giảm loss là task score.
