# Huấn luyện Qwen3.5-4B QLoRA một epoch (có resume)

Tích hợp từ các thử nghiệm 016 (wrapper Qwen3.5) và 020 (trainer một epoch) đã tạo adapter checkpoint 0880 của pipeline ASR+LLM. Logic huấn luyện giữ nguyên; các đường dẫn cứng của workspace thành tham số. Mã nguồn gốc nằm ở tag `archive/experiments-011-021` của repo workspace.

## Thành phần

| Module | Vai trò |
|---|---|
| `training/qwen35.py` | Tokenizer Qwen3.5 luôn render với `enable_thinking=False` (khớp inference); `load_base` NF4 cho model `qwen3_5`/`qwen3_5_text`, từ chối nếu có vision tower; kernel SDPA EFFICIENT+FLASH (EFFICIENT lỗi với `enable_gqa=True` khi không có padding mask). |
| `training/epoch_schedule.py` | Thứ tự một lượt (hàng dài nhất đầu tiên), nhóm accumulation (nhóm cuối có thể thiếu), warmup + cosine theo chỉ số bước. |
| `training/train_epoch.py` | Trainer: LoRA rank 16/alpha 32/all-linear, loss hành động chính xác trên logits chọn lọc, checkpoint đầy đủ (adapter, optimizer, RNG, con trỏ dữ liệu, hash), resume có xác minh, kiểm tra reload adapter. Lệnh `lifemate train-epoch`. |
| `training/verify_resume.py` | Smoke ba tiến trình: chạy liền, dừng ở checkpoint 1, resume; so adapter, optimizer, RNG, scheduler. Lệnh `lifemate verify-resume`. |
| `training/prepare_trajectories.py --inner-split` | Dựng dataset từ một inner split cố định (không suy ra từ seed). Tokenizer Qwen3.5 tự được patch non-thinking; Qwen3 không đổi. |

## Môi trường

Python 3.12, Torch 2.8.0+cu128, Transformers 5.22.1, PEFT 0.21.2, bitsandbytes 0.50.2, flash-linear-attention 0.5.0; cài theo `requirements-qwen35-training-linux-py312.lock.txt` (lệnh cài ở đầu file). Môi trường baseline cũ không import được Qwen3.5. GPU BF16 bắt buộc, cần ít nhất 20 GiB đĩa trống cho checkpoint đầy đủ.

## Tái lập adapter 0880

Chạy từ thư mục `lifemate-agent/` với `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:.`

```bash
# 1. Dataset: 1.112 fit / 124 validation, 3.517 / 392 hàng (split lịch sử khôi phục từ run 0880)
python -m lifemate.cli prepare-trajectories --data-root ../Data --manifest <split_manifest.json> \
    --model-path ../Models/Qwen3.5-4B --output-dir <dataset> --inner-split <inner_split.json>
python -m lifemate.cli audit-trajectories --dataset-dir <dataset> --model-path ../Models/Qwen3.5-4B \
    --output <dataset>/audit.json --torch-probe

# 2. Kiểm tra cấu hình, không huấn luyện
python -m lifemate.cli train-epoch --dataset-dir <dataset> --model-path ../Models/Qwen3.5-4B --output-dir <run> --preflight

# 3. Huấn luyện một epoch: 880 bước, accumulation 4, lr 1e-4, seed 42, checkpoint mỗi 100 bước
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m lifemate.cli train-epoch \
    --dataset-dir <dataset> --model-path ../Models/Qwen3.5-4B --output-dir <run> --seed 42
# Resume sau khi bị dừng: thêm --resume <run>/checkpoint-0NNN, output-dir mới
```

`--max-train-seconds`, `--stop-after` và `--save-every` cho phép dừng có checkpoint. `--smoke-rows 9 --validation-limit 4 --save-every 1` là smoke (hàng dài nhất cộng các hàng ngắn).

Lệnh cụ thể đã dùng để xác minh việc tích hợp (dựng lại dataset từ split lịch sử, đối chiếu với run 0880, smoke resume) nằm ở `Experiments/024-qwen35-training-integration/README.md` của repo workspace.

## Giới hạn

- Checkpoint tạo bằng phiên bản thử nghiệm cũ (kể cả 0680/0700/0800/0880) **dùng được để inference nhưng không resume được** bằng bản này: vân tay cấu hình gồm hash mã nguồn và hash `dataset_manifest.json`, cả hai đã đổi.
- Training GPU không nhất thiết bit-identical giữa các máy; adapter 0880 gốc huấn luyện trên GPU khác. Việc tái tạo thứ tự dữ liệu được xác minh bằng so sánh; việc tái tạo trọng số bằng huấn luyện lại đầy đủ (khoảng 7 giờ) chưa được thực hiện.
- Dữ liệu là teacher-forced prefix, không phải đánh giá closed-loop.
