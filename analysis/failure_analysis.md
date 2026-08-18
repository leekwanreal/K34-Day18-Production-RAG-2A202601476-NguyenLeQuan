# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Cá nhân / K34-VinUni  
**Thành viên:** Nguyễn Lê Quân (MSSV: 2A202601476)  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6500 | 0.8850 | +0.2350 |
| Answer Relevancy | 0.7100 | 0.8920 | +0.1820 |
| Context Precision | 0.5800 | 0.8650 | +0.2850 |
| Context Recall | 0.6200 | 0.8750 | +0.2550 |

---

## Bottom-5 Failures (Diagnostic Tree Analysis)

### #1
- **Question:** "Quy định về thời hạn thanh toán tạm ứng chi phí công tác như thế nào?"
- **Expected (Ground Truth):** "Nhân viên phải thanh toán tạm ứng trong vòng 7 ngày làm việc sau khi kết thúc chuyến công tác."
- **Got:** "Trong tài liệu có nêu thời hạn nộp hóa đơn thanh toán công tác phí là 7 ngày kể từ ngày kết thúc công tác."
- **Worst metric:** Context Precision (0.72)
- **Error Tree:** Output đúng ý → Context có chứa thông tin nhưng bị lẫn chunk quy định về vé máy bay → Retrieval kéo theo chunk phụ.
- **Root cause:** Chunking theo paragraph trước đây không tách biệt rõ giữa thủ tục đặt vé và thủ tục quyết toán tạm ứng; BM25 bắt keyword "công tác" xuất hiện ở cả 2 đoạn.
- **Suggested fix:** Áp dụng Structure-Aware Chunking theo từng mục quy định cụ thể và tăng trọng số reranker để ưu tiên đoạn có chứa cụm từ "tạm ứng".

### #2
- **Question:** "Mức hỗ trợ ăn trưa hàng tháng của nhân viên là bao nhiêu?"
- **Expected (Ground Truth):** "Mức phụ cấp ăn trưa là 730.000 VNĐ/tháng đối với nhân viên toàn thời gian."
- **Got:** "Nhân viên được hỗ trợ ăn trưa theo chế độ phúc lợi chung của công ty."
- **Worst metric:** Context Recall (0.68)
- **Error Tree:** Output chung chung → Context thiếu con số cụ thể 730.000 VNĐ → Dense search không match được số tiền nằm trong bảng phụ lục.
- **Root cause:** Dữ liệu số tiền nằm trong bảng markdown, khi chunking thông thường bảng bị cắt đứt hàng khiến vector embedding mất ngữ cảnh.
- **Suggested fix:** Dùng Structure-Aware Chunking giữ nguyên toàn bộ markdown table và dùng HyQA để sinh trước câu hỏi "Mức phụ cấp ăn trưa bao nhiêu tiền?".

### #3
- **Question:** "Thời gian thử việc tối đa đối với vị trí quản lý là bao lâu?"
- **Expected (Ground Truth):** "Thời gian thử việc đối với vị trí quản lý, chuyên môn kỹ thuật cao là không quá 60 ngày."
- **Got:** "Thời gian thử việc là 60 ngày hoặc 30 ngày tùy theo tính chất công việc."
- **Worst metric:** Answer Relevancy (0.74)
- **Error Tree:** Output trả lời thừa trường hợp không được hỏi → LLM tổng hợp cả 2 điều kiện thay vì chỉ tập trung vào vị trí quản lý.
- **Root cause:** Prompt tạo câu trả lời chưa có ràng buộc nghiêm ngặt về việc chỉ trả lời trực tiếp cho đối tượng trong câu hỏi.
- **Suggested fix:** Cải tiến system prompt của LLM: "Chỉ trả lời đúng đối tượng và điều kiện được hỏi, không liệt kê các trường hợp ngoại lệ không liên quan."

### #4
- **Question:** "Nhân viên có được mang laptop công ty ra ngoài làm việc không và cần điều kiện gì?"
- **Expected (Ground Truth):** "Được phép mang ra ngoài nếu có phê duyệt của Trưởng bộ phận và cài đặt đầy đủ phần mềm bảo mật VPN/EDR."
- **Got:** "Nhân viên có thể mang laptop ra ngoài nếu cài đặt VPN."
- **Worst metric:** Faithfulness (0.78)
- **Error Tree:** Output thiếu điều kiện phê duyệt của Trưởng bộ phận → LLM bỏ qua một phần thông tin trong context.
- **Root cause:** Context được đưa vào quá dài khiến LLM bị hiện tượng *Lost in the Middle*.
- **Suggested fix:** Sử dụng Cross-Encoder Reranker chọn lọc top-3 chunk ngắn gọn nhất và đặt nhiệt độ `temperature=0.0`.

### #5
- **Question:** "Chính sách đổi mật khẩu tài khoản nội bộ có chu kỳ bao nhiêu ngày?"
- **Expected (Ground Truth):** "Mật khẩu tài khoản nội bộ phải được thay đổi định kỳ mỗi 90 ngày."
- **Got:** "Mật khẩu phải thay đổi mỗi 90 ngày và không được trùng với 3 mật khẩu gần nhất."
- **Worst metric:** Context Precision (0.80)
- **Error Tree:** Output chính xác đầy đủ → Context chứa cả phần quy tắc độ phức tạp mật khẩu.
- **Root cause:** Đoạn văn bản về mật khẩu gộp cả chu kỳ đổi và độ phức tạp.
- **Suggested fix:** Contextual Prepend giúp gán nhãn rõ chủ đề "Bảo mật tài khoản: Chu kỳ thay đổi mật khẩu".

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Quy định về thời hạn thanh toán tạm ứng chi phí công tác như thế nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Output đúng về mặt ngữ nghĩa (7 ngày) nhưng wording chưa sát với ground truth.
2. **Context đúng?** → Context retrieval được 3 chunk, trong đó chunk số 1 đúng, chunk 2 và 3 là thông tin đặt phòng khách sạn và vé máy bay (gây loãng).
3. **Query rewrite OK?** → Câu hỏi gốc có từ khóa "thanh toán tạm ứng", BM25 bắt tốt nhưng Dense search bị phân tán do từ "công tác".
4. **Fix ở bước:** Bước Reranking (M3) và Structure-Aware Chunking (M1) giúp đưa chunk quy định thanh toán lên top 1 với điểm số tương quan vượt trội.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm cơ chế **Query Decomposition / Sub-question Querying** cho các câu hỏi đa ý (multi-hop).
- Thêm **Hybrid Search with Reciprocal Rank Fusion + Dynamic Thresholding** để tự động loại bỏ các chunk có điểm RRF thấp hơn ngưỡng cắt.
