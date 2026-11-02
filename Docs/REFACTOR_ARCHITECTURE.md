# Refactor theo kiến trúc hệ thống

Ngày: 02/11/2026. Cấu trúc đích theo mục 14 của `lifemate-workspace/Docs/system_architecture.md`. Source trước refactor: commit `996f13e`; bản ghi baseline lịch sử vẫn giữ nguyên, không thay bằng số đo của lần refactor.

## Mã hiện tại

```text
src/lifemate/
  cli.py
  contracts.py
  input/adapter.py
  context/manager.py
  agent/{prompt,planner,controller}.py
  tools/env_adapter.py
  output/writer.py
training/{preparation,prepare_data}.py
evaluation/{replay,replay_train,analysis,run_text_baseline}.py
tests/{unit,contract,integration}/
```

`Docs`, `Assets` và các file requirements giữ tên/vị trí hiện có. Package distribution tên `lifemate-agent` (trước là `lifemate-data`). Không thêm thư mục ASR, config YAML, LoRA hoặc Docker trống; đó là phần triển khai tiếp theo.

## Ánh xạ từ cấu trúc trước

| Trước | Sau |
|---|---|
| `lifemate_data.loader` | `lifemate.input.adapter`; dataclasses sang `lifemate.contracts` |
| `lifemate_data.preparation` | `training.preparation` |
| `lifemate_tools.adapter` | `lifemate.tools.env_adapter` |
| `lifemate_agent.text` | `lifemate.context.manager`, `lifemate.agent.prompt`, `.planner`, `.controller` |
| `lifemate_agent.local_model` | `lifemate.agent.planner.LocalPlanner` |
| `lifemate_output.writer` | `lifemate.output.writer` |
| `lifemate_evaluation.replay` | `evaluation.replay` |
| `lifemate_evaluation.analysis` | `evaluation.analysis` |
| `scripts/prepare_data.py` | `training/prepare_data.py`, script cũ là wrapper |
| `scripts/replay_train.py`, `scripts/run_text_baseline.py` | Runner trong `evaluation`, script cũ là wrapper |

Các package `lifemate_*` ở `src` chỉ là import aliases, không giữ bản sao implementation hoặc registry ToolEnv riêng. Alias và canonical path trả cùng class object. Điều này giữ các import công khai đã dùng bởi snapshots Experiments. Các monkey-patch vào chi tiết private của module cũ không phải API tương thích; tests hiện tại patch đường dẫn canonical.

## Chạy từ source và từ package

Source, từ thư mục repo:

```powershell
$env:PYTHONPATH = "$(Join-Path (Get-Location) 'src');$(Get-Location)"
.venv/Scripts/python.exe -m lifemate.cli --help
.venv/Scripts/python.exe -m lifemate.cli prepare-data --help
.venv/Scripts/python.exe -m lifemate.cli replay-train --help
.venv/Scripts/python.exe -m lifemate.cli text-baseline --help
.venv/Scripts/python.exe -m lifemate.cli analyze-run --help
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Đặt cả `src` và repo root trên PYTHONPATH vì `training` và `evaluation` là các package ở root. Khi chạy snapshot cũ từ thư mục khác, dùng cùng hai đường dẫn tuyệt đối hoặc cài package trong môi trường chạy snapshot. Không cần sửa snapshot.

Editable install trong môi trường đã có pip:

```powershell
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/lifemate.exe --help
```

Hoặc `uv pip install --python .venv/Scripts/python.exe -e .`. Khi đã cài, có thể dùng module CLI từ thư mục khác mà không tự đặt PYTHONPATH. Chỉ cài công cụ, không tải model hoặc gọi mạng inference. Các script cũ vẫn nhận arguments như trước; entry point download model vẫn là script phát triển riêng.

Ví dụ baseline mới, thêm nguyên arguments đã dùng trước đây sau tên workflow:

```powershell
.venv/Scripts/python.exe -m lifemate.cli text-baseline --data-root ../Data --manifest ../Outputs/004-train-dev-split/run-20261102-01/split_manifest.json --model-path ../Models/Qwen3-4B-Instruct-2507 --output-dir ../Outputs/005-text-baseline/run-new --limit 3
```

## Hành vi được giữ nguyên

- Prompt fingerprint, tokenization/messages, parsing và vòng controller.
- BF16, greedy, seed, context/token/call/repair budgets, chunked prefill và decode.
- Contract dữ liệu, một ToolEnv/mẫu, ledger/hash và Writer.
- Manifest, trọng số và các output baseline lịch sử.

Không chuyển inference sang evaluator; gold/reference vẫn chỉ ở phía đánh giá. Không chạy lại 309 mẫu LLM hoặc điều chỉnh chất lượng model trong bước tổ chức source.

## Bằng chứng

Tính từ workspace root: `Outputs/refactor-architecture/run-20261102-01/`.

- `tests.txt`: **49/49 tests đạt**, gồm 47 tests đã có và 2 kiểm tra alias/contracts/prompt.
- `behavior-ast.json`: 9 định nghĩa lõi (controller, parser, context, messages, LocalPlanner, loader, adapter, Writer/checks) có AST giống commit baseline trước refactor, bỏ qua vị trí file.
- `smoke/report.json`: 3 mẫu finish-only qua CLI mới; đây là plumbing, không phải LLM quality.
- `replay/report.json`: 1.545/1.545 train khớp state/call/results, 3.541 calls.
- `verification.json`: analysis qua CLI mới bằng báo cáo gốc (109 hash matches, 82 raw-state, 30 exact calls); wheel import và bốn workflow CLI hoạt động từ thư mục giải nén độc lập source.
- Wheel đóng gói có `lifemate`, `training`, `evaluation` và các shim tương thích; không có dữ liệu/model/outputs.

Snapshots `Experiments` được giữ nguyên. Mục 14 đã cập nhật để mô tả cấu trúc thật, phân biệt phần đang có với kế hoạch tương lai. Việc commit được thực hiện riêng theo yêu cầu sau khi kiểm tra; không tự push lên remote.
