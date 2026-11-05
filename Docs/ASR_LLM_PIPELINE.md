# Pipeline ASR + LLM: từ audio đến submission

Một lệnh biến audio của một split test thành `submission.jsonl` hợp lệ:

```text
audio lượt cuối + mọi audio user trong history
  → PhoWhisper-medium LoRA (CTranslate2 fp16)         lifemate transcribe
  → Qwen3.5-4B + adapter LoRA (merge BF16), controller  lifemate run-agent
  → ToolEnv của BTC (một ToolEnv cho mỗi mẫu)
  → submission.jsonl → python -m zaic_tools --validate   lifemate predict (chạy cả ba bước)
```

Tổng số tham số khoảng 4,2B (LLM đã merge) cộng khoảng 0,77B (ASR), dưới giới hạn 6B của cuộc thi.

## Môi trường

Hai giai đoạn dùng hai bộ phụ thuộc khác nhau nên thường là hai môi trường; `lifemate predict` nhận `--asr-python` và `--llm-python`. Nếu một môi trường có đủ cả hai bộ thì bỏ qua cả hai tham số.

| Giai đoạn | Cài đặt | Phiên bản đã kiểm chứng |
|---|---|---|
| ASR | `pip install "lifemate-agent[asr]"` | faster-whisper 1.2.1, ctranslate2 4.8.5, PyAV < 15 (PyAV 15 bỏ tham số `metadata_errors` mà faster-whisper 1.2.1 dùng) |
| LLM | `pip install "lifemate-agent[llm]"` hoặc `requirements-qwen35-training-linux-py312.lock.txt` | Torch 2.8.0+cu128, Transformers 5.22.1, PEFT 0.21.2, flash-linear-attention 0.5.0 |

Phần lõi (loader, ToolEnv adapter, controller, scorer, submission writer) chỉ dùng thư viện chuẩn. Weights không thuộc repo: cần thư mục Qwen3.5-4B, adapter LoRA và model ASR đã chuyển sang CTranslate2 float16.

## Chạy

```bash
lifemate predict --split public_test --data-root <Data> \
    --asr-model <phowhisper-ct2-fp16> --llm-model <Qwen3.5-4B> --adapter <adapter-dir> \
    --asr-python <env-asr>/bin/python --llm-python <env-llm>/bin/python \
    --output-dir <thư-mục-mới>
# submission: <thư-mục-mới>/agent/inference/submission.jsonl
```

`--ids-file` chạy trên một tập con để smoke; một submission thật cần toàn bộ split. Từng bước chạy riêng được: `lifemate transcribe …` ghi `transcripts.jsonl`, `lifemate run-agent --mode asr --transcripts …` đọc nó.

## Đảm bảo

- ASR chỉ đọc đầu vào của agent; không bao giờ lấy transcript chuẩn. Hypothesis rỗng được giữ nguyên và đếm, không có dự phòng.
- `run-agent` ghi `inputs_used.jsonl`: hash của văn bản thực sự đưa vào planner, để kiểm tra sau.
- Mỗi mẫu một ToolEnv; history không bao giờ chạy lại. Trace bền vững được audit trước khi báo kết quả; audit không hợp lệ thì lệnh thất bại.
- Chế độ `--mode oracle` (transcript chuẩn) chỉ cho phần train/dev và bị từ chối trên split test.
- Ngân sách mỗi mẫu: 30 giây, 8 lệnh gọi, 12 quyết định, 2 lần sửa.

## Đánh giá trên dev

`lifemate run-agent --split train --manifest <split_manifest.json> --ids-file <dev-ids> …` chạy bộ đánh giá của repo (replay độc lập, trace, điểm local) trên phân vùng dev logic của dữ liệu train. `--manifest-sha256` từ chối chạy nếu manifest đổi. Điểm là điểm local, chưa được BTC chứng nhận.

## Giới hạn

- ASR dùng profile `guarded` (nhiệt độ dự phòng có thể lấy mẫu) nên không bit-reproducible; số audio phải dự phòng được ghi trong `meta.json`.
- Vòng lặp sản xuất được test trên CPU với planner và ToolEnv giả; đường GPU thật được kiểm bằng smoke.
- Đường dẫn model là tham số; repo không chứa weights hay dữ liệu.
