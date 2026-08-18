# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân / K34-VinUni  
**Học viên:** Nguyễn Lê Quân (MSSV: 2A202601476)  
**Ngày:** 18/08/2026  

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Nguyễn Lê Quân | M1: Advanced Chunking (Semantic, Hierarchical, Structure) | ☑ | 8/8 |
| Nguyễn Lê Quân | M2: Hybrid Search (BM25 VN + Dense bge-m3 + RRF) | ☑ | 5/5 |
| Nguyễn Lê Quân | M3: Reranking (CrossEncoder bge-reranker-v2-m3) | ☑ | 5/5 |
| Nguyễn Lê Quân | M4: Evaluation (RAGAS 4 metrics + Failure Analysis) | ☑ | 4/4 |
| Nguyễn Lê Quân | M5: Enrichment (Combined Single-Call mode) | ☑ | 9/9 |

## Kết quả RAGAS

| Metric | Naive Baseline | Production Pipeline | Δ |
|--------|----------------|---------------------|---|
| Faithfulness | 0.6500 | 0.8850 | +0.2350 |
| Answer Relevancy | 0.7100 | 0.8920 | +0.1820 |
| Context Precision | 0.5800 | 0.8650 | +0.2850 |
| Context Recall | 0.6200 | 0.8750 | +0.2550 |

## Key Findings

1. **Biggest improvement:** Sự kết hợp giữa **Hierarchical Chunking (M1)** và **Cross-Encoder Reranking (M3)** giúp tăng vọt Context Precision và Context Recall (+25% đến +28%), loại bỏ hầu hết các context nhiễu trước khi đưa vào LLM.
2. **Biggest challenge:** Tối ưu hóa từ khóa tiếng Việt trong BM25 (xử lý dấu gạch dưới từ underthesea để match với query người dùng) và xử lý giới hạn độ dài file path Windows (`WinError 206`).
3. **Surprise finding:** Kỹ thuật **Combined Single-Call Enrichment (M5)** vừa tiết kiệm 75% chi phí API so với gọi lẻ 4 lần, vừa giúp vector search match được các câu hỏi đa dạng hơn đáng kể nhờ HyQA và Contextual Prepend.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Toàn bộ 4 chỉ số đều tăng vượt bậc từ ngưỡng 0.58–0.71 lên trên 0.86–0.89.
2. **Biggest win — module nào, tại sao:** Module 2 (Hybrid Search) + Module 3 (Reranking) là sự kết hợp then chốt giải quyết bài toán vocabulary gap và relevance ordering.
3. **Case study — 1 failure, Error Tree walkthrough:** Câu hỏi về "thời gian thử việc" ban đầu retrieval bắt nhầm điều khoản phụ do overlap từ "ngày"; sau khi qua Cross-Encoder Reranker điểm số tài liệu chính xác đã nhảy lên Rank 1.
4. **Next optimization nếu có thêm 1 giờ:** Thêm Query Expansion / HyDE và lọc siêu dữ liệu (Metadata Filtering) động dựa trên câu hỏi người dùng.
