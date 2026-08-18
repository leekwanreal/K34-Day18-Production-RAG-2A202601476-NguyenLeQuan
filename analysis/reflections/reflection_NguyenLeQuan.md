# Reflection: Lecture → Project — Lab 18: Production RAG Pipeline

**Học viên:** Nguyễn Lê Quân  
**MSSV:** 2A202601476  
**Lớp / Khóa:** K34  
**Ngày thực hiện:** 18/08/2026  

---

## Phần 1: Mapping bài giảng vào Code (Lecture Concepts → Code Implementation)

| Lecture Concept | Module | Hàm / Class cụ thể | Observation & Phân tích chuyên sâu |
|---|---|---|---|
| **Semantic Chunking** | M1: Chunking | `chunk_semantic()` | Dùng `SentenceTransformer("all-MiniLM-L6-v2")` encode từng câu, tính cosine similarity giữa các câu liền kề với ngưỡng threshold `0.85`. Giúp gom các câu cùng chủ đề vào 1 chunk, tránh cắt ngang ngữ nghĩa như basic chunking. |
| **Hierarchical Chunking (Parent-Child)** | M1: Chunking | `chunk_hierarchical()` | Tạo Parent Chunk (2048 chars) lưu toàn bộ ngữ cảnh lớn, tách thành Child Chunks (256 chars) để vector search với độ chính xác cao (high precision retrieval) nhưng khi trả về cho LLM thì có thể mở rộng ngữ cảnh từ parent. |
| **Structure-Aware Chunking** | M1: Chunking | `chunk_structure_aware()` | Dùng Regex `(^#{1,3}\s+.+$)` nhận diện cấu trúc Markdown (H1, H2, H3) và gán `section` vào metadata, giữ nguyên bảng biểu, bullet point và code block liền mạch. |
| **Vietnamese Tokenization & BM25** | M2: Hybrid Search | `segment_vietnamese()`, `BM25Search` | `underthesea` tách từ ghép tiếng Việt, xử lý thay thế `_` thành space để tương thích bộ tách từ của `BM25Okapi`. Giúp bắt chính xác từ khóa chuyên ngành, số liệu, mã quy định. |
| **Dense Vector Search** | M2: Hybrid Search | `DenseSearch` | Dùng model đa ngôn ngữ `BAAI/bge-m3` (1024 dim) kết hợp cơ sở dữ liệu vector Qdrant (`query_points()`) để nắm bắt ngữ nghĩa sâu và tìm kiếm tương đồng đa ngữ. |
| **Reciprocal Rank Fusion (RRF)** | M2: Hybrid Search | `reciprocal_rank_fusion()` | Hợp nhất danh sách xếp hạng từ BM25 và Dense Search theo công thức $RRF\_Score(d) = \sum \frac{1}{k + rank + 1}$ với $k=60$, cân bằng giữa tìm kiếm từ khóa chính xác và tìm kiếm ngữ nghĩa mà không cần chuẩn hóa scale điểm. |
| **Cross-Encoder Reranking** | M3: Reranking | `CrossEncoderReranker.rerank()` | Sử dụng model `BAAI/bge-reranker-v2-m3` nhận trực tiếp cặp `(query, document)` để tính điểm tương quan chéo (cross-attention). Rerank từ top-20 xuống top-3 giúp lọc sạch 80%+ tài liệu nhiễu trước khi đưa vào LLM context. |
| **RAGAS Evaluation 4 Metrics** | M4: Evaluation | `evaluate_ragas()`, `failure_analysis()` | Đánh giá toàn diện 4 chiều: *Faithfulness* (chống hallucination), *Answer Relevancy* (đúng trọng tâm câu hỏi), *Context Precision* (tỷ lệ chunk liên quan ở thứ hạng cao), *Context Recall* (đầy đủ thông tin ground truth). Tích hợp Diagnostic Tree để tự động chẩn đoán lỗi. |
| **Enrichment (HyQA + Contextual Prepend + Summary)** | M5: Enrichment | `_enrich_single_call()`, `enrich_chunks()` | Tối ưu chi phí bằng 1 Single LLM Call (GPT-4o-mini) cho mỗi chunk để vừa tạo summary, sinh 3 câu hỏi giả định (HyQA), Contextual Prepend (giảm 49% retrieval failure theo Anthropic) và gán auto metadata. |

---

## Phần 2: Khó khăn gặp phải & Cách giải quyết

### 1. Lỗi đường dẫn dài trên Windows (`[WinError 206]`)
- **Exact Error Message:**  
  `ERROR: Could not install packages due to an OSError: [WinError 206] The filename or extension is too long: 'C:\Users\Admin\Desktop\Nus 20252\VinUni\Lab\K34-Day18-Production-RAG-2A202601476-NguyenLeQuan\.venv\Lib\site-packages\...'`
- **Nguyên nhân:** Windows mặc định áp dụng giới hạn đường dẫn tối đa 260 ký tự (`MAX_PATH`). Khi `pip install` các thư viện lớn có cây thư mục sâu như PyTorch, Sentence-Transformers, Ragas vào thư mục con của `.venv`, độ dài path vượt quá 260 ký tự.
- **Cách giải quyết:** Mở PowerShell với quyền Administrator và bật cờ `LongPathsEnabled` trong Windows Registry:
  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```
  Sau đó khởi động lại terminal và chạy `pip install -r requirements.txt` thành công 100%.

### 2. Sự không tương thích giữa Tokenizer tiếng Việt (`underthesea`) và `BM25Okapi`
- **Vấn đề:** `underthesea.word_tokenize(text, format="text")` nối từ ghép bằng dấu gạch dưới (ví dụ: `nghỉ_phép`). Khi BM25 split bằng khoảng trắng, câu truy vấn của người dùng `"nghỉ phép"` tách thành 2 token `["nghỉ", "phép"]` trong khi corpus là `["nghỉ_phép"]` dẫn đến không khớp từ khóa.
- **Cách giải quyết:** Xử lý post-processing `.replace("_", " ")` sau khi tách từ, đảm bảo token thống nhất ở cả bước Indexing và Search.

### 3. Tối ưu chi phí & Latency của Module Enrichment
- **Vấn đề:** Nếu gọi 4 API calls riêng lẻ (Summarize, HyQA, Contextual, Metadata) cho 100 chunks, hệ thống sẽ tốn 400 LLM calls, gây chậm và tốn token API.
- **Cách giải quyết:** Viết hàm `_enrich_single_call()` gom toàn bộ 4 nhiệm vụ vào 1 structured prompt duy nhất trả về định dạng JSON, vừa giảm 75% số lượng request, vừa đảm bảo tốc độ và nhận điểm bonus +2 của bài lab.

---

## Phần 3: Action Plan áp dụng vào Project cá nhân

### Thông tin Project
- **Tên project:** Hệ thống Trợ lý RAG Tra cứu Quy chế và Tài liệu Nội bộ Doanh nghiệp
- **Hiện trạng pipeline:** Đang sử dụng chunking cơ bản theo ký tự cố định (fixed-size 500 chars), tìm kiếm thuần vector (dense-only bằng OpenAI embeddings) không có Reranking hay Enrichment.
- **Known Issues:**
  - Tra cứu các điều khoản số hiệu (ví dụ: "Điều 15 khoản 2", "mức phạt 5.000.000 VNĐ") thường bị bỏ sót do dense search không match chính xác keyword.
  - LLM thỉnh thoảng hallucinate khi context chứa quá nhiều thông tin rác ngoài lề.

### Kế hoạch áp dụng các kỹ thuật từ Lab 18
1. **Chunking Strategy:** Chuyển sang kết hợp **Structure-Aware Chunking** (bóc tách theo cấu trúc Điều/Khoản của văn bản quy phạm) và **Hierarchical Chunking** (Parent 2000 chars, Child 300 chars) để đảm bảo ngữ cảnh đầy đủ khi trả lời.
2. **Search Strategy:** Chuyển đổi từ Dense-only sang **Hybrid Search (BM25 tiếng Việt + bge-m3 dense + RRF)** để khắc phục triệt để lỗi tra cứu từ khóa và mã hiệu.
3. **Reranking:** Tích hợp `BAAI/bge-reranker-v2-m3` để lọc Top 25 retrieval còn Top 5 tài liệu sát nhất trước khi gửi prompt vào LLM.
4. **Evaluation:** Thiết lập bộ benchmark tự động bằng **RAGAS** với 50 câu hỏi synthetic ground-truth để đo lường định kỳ mỗi khi cập nhật dữ liệu.
5. **Enrichment:** Áp dụng **Contextual Prepend** để bổ sung nguồn gốc văn bản vào đầu mỗi chunk trước khi index vào Qdrant.

### Timeline thực hiện
- **Tuần 1:** Chuẩn hóa dữ liệu văn bản nội bộ, implement Structure-Aware & Hierarchical Chunking.
- **Tuần 2:** Setup Qdrant Vector DB, dựng Hybrid Search với BM25 và RRF.
- **Tuần 3:** Tích hợp Cross-Encoder Reranker, tinh chỉnh prompt LLM.
- **Tuần 4:** Chạy RAGAS benchmark, tối ưu các case điểm thấp qua Failure Analysis và release production.
