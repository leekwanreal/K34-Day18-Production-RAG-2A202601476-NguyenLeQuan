from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


_CROSS_ENCODER_CACHE = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name

    def _load_model(self):
        global _CROSS_ENCODER_CACHE
        if self.model_name not in _CROSS_ENCODER_CACHE:
            try:
                from sentence_transformers import CrossEncoder
                _CROSS_ENCODER_CACHE[self.model_name] = CrossEncoder(self.model_name, local_files_only=True, trust_remote_code=True)
            except Exception:
                try:
                    from flashrank import Ranker
                    _CROSS_ENCODER_CACHE[self.model_name] = FlashrankReranker()
                except Exception:
                    from sentence_transformers import SentenceTransformer
                    _CROSS_ENCODER_CACHE[self.model_name] = SentenceTransformer("all-MiniLM-L6-v2")
        return _CROSS_ENCODER_CACHE[self.model_name]

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        if isinstance(model, FlashrankReranker):
            return model.rerank(query, documents, top_k=top_k)
        try:
            from sentence_transformers import CrossEncoder
            if isinstance(model, CrossEncoder):
                pairs = [(query, doc["text"]) for doc in documents]
                scores = model.predict(pairs)
                if hasattr(scores, "tolist"):
                    scores = scores.tolist()
                elif isinstance(scores, (int, float)):
                    scores = [scores]
            else:
                from numpy import dot
                from numpy.linalg import norm
                q_vec = model.encode(query)
                doc_vecs = model.encode([d["text"] for d in documents])
                scores = [float(dot(q_vec, dv) / (norm(q_vec) * norm(dv) + 1e-9)) for dv in doc_vecs]

            scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}),
                    rank=i
                )
                for i, (score, doc) in enumerate(scored[:top_k])
            ]
        except Exception as e:
            print(f"  ⚠️ Reranker fallback: {e}")
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {}),
                    rank=i
                )
                for i, doc in enumerate(documents[:top_k])
            ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            passages = [{"id": i, "text": d["text"], "meta": d.get("metadata", {})} for i, d in enumerate(documents)]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            return [
                RerankResult(
                    text=r["text"],
                    original_score=documents[r["id"]].get("score", 0.0),
                    rerank_score=float(r["score"]),
                    metadata=documents[r["id"]].get("metadata", {}),
                    rank=i
                )
                for i, r in enumerate(results[:top_k])
            ]
        except Exception as e:
            print(f"  ⚠️ Flashrank rerank failed: {e}")
            return [
                RerankResult(
                    text=d["text"],
                    original_score=d.get("score", 0.0),
                    rerank_score=0.0,
                    metadata=d.get("metadata", {}),
                    rank=i
                )
                for i, d in enumerate(documents[:top_k])
            ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
