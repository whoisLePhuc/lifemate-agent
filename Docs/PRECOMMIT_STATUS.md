# Trạng thái chuẩn bị commit

Ngày kiểm tra: 02/11/2026. Chưa stage, commit hoặc push thay đổi trong bước chuẩn bị này.

## Các mục đã hoàn thiện

- Công cụ phân tích nằm trong `src/lifemate_evaluation/analysis.py`, entry point `scripts/analyze_run.py`; repo agent không cần import mã từ Experiments. Wrapper cũ chỉ giữ cho tương thích workspace.
- Thêm 6 tests về matching one-to-one, phân biệt tool/arguments, ID thiếu/trùng, output không ghi đè/thoát thư mục và gold-hash replay đúng một session mỗi mẫu.
- [Hướng dẫn baseline](TEXT_BASELINE.md), [phân tích](BASELINE_ANALYSIS.md), README và kế hoạch workspace phản ánh run 04 đã hoàn tất. Lịch sử tải lỗi/run dừng nằm trong phần lịch sử thử nghiệm, không còn được mô tả là tiến trình đang chạy.
- [Tóm tắt aggregate](Results/text-baseline-v0.1.json) lưu model revision, fingerprints, phiên bản môi trường, cấu hình, lệnh chạy và số liệu 309 dev; không chứa sample IDs, transcript, arguments, app state hoặc đường dẫn cá nhân.
- [Dependency pins](../requirements-baseline-windows-py312.lock.txt) khớp venv đã dùng; requirements trực tiếp pin rõ CUDA build. Đây là snapshot Windows/Python 3.12, chưa phải lock cho Docker/A30.
- `download_ranges.py` giữ ở Experiments làm công cụ xử lý sự cố; không đưa vào inference hoặc package agent.

## Bằng chứng kiểm tra

Tính từ workspace root, `Outputs/precommit-verification/run-20261102-01/`:

- `tests.txt`: **47/47 tests đạt**, gồm official-runtime opt-in; không chạy lại LLM 309 mẫu.
- `checks.json`: 7 CLI help thành công, phân tích tích hợp bằng báo cáo gốc, artifact fingerprints khớp, tóm tắt không có bản ghi mẫu/đường dẫn cá nhân, pins khớp venv.
- `installed-requirements.txt`: snapshot môi trường để đối chiếu.

Phân tích chạy lại ở `Outputs/005-text-baseline/run-20261102-04-full-dev/analysis-precommit-verified.json` cho cùng số liệu: **109 hash matches / 82 raw-state matches / 30 exact full-call matches**. Gold replay trong lượt evaluator riêng, không sửa prediction hoặc mở lại session inference.

Kiểm tra 16 tài liệu Markdown (trước khi thêm trang trạng thái này): không link local hỏng hoặc fence không cân bằng. `git diff --check` đạt; file drawio của workspace có thông báo chuyển đổi LF/CRLF theo cấu hình Git. Không sửa sơ đồ trong bước này. Toolkit BTC có ResourceWarning về file handle khi test bật warning; tests vẫn đạt và toolkit không bị sửa.

## Giới hạn cần giữ khi viết commit

Run được thực hiện trước commit, nên `source_commit` trong summary để null. Không gán SHA của HEAD cũ làm bằng chứng mã đã chạy. Sau commit có thể gắn SHA mới như mốc lưu trữ baseline, kèm mô tả đúng provenance.

109/309 là tỷ lệ hash match; 82/309 là raw-state equality; 30/309 là exact call diagnostic. Không dùng chúng như điểm BTC: chưa có fuzzy token-F1/official scorer. Các bước ASR, fine-tuning và Docker chấm thi chưa hoàn thành.

Data, Models, Outputs, .venv và caches vẫn ngoài Git. Không đưa submission/journal/raw logs vào commit; chỉ lưu aggregates và tài liệu tái lập. Các commit chức năng tiếp theo thực hiện riêng ở hai repo, không gom repo con vào Git workspace.
