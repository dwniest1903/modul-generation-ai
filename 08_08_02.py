import numpy as np
import math
from collections import Counter
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
EMBED_MODEL = "nomic-embed-text"

@dataclass
class HybridResult:
    doc_id: str
    text: str
    final_score: float
    semantic_score: float
    keyword_score: float

class BM25Scorer:
    """Implementasi sederhana BM25 Keyword Search."""
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [text.lower().split() for text in corpus]
        self.doc_count = len(corpus)
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_doc_len = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0
        
        # Hitung Document Frequency (DF)
        self.df = Counter()
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.df[term] += 1

    def score(self, query: str) -> np.ndarray:
        query_terms = query.lower().split()
        scores = np.zeros(self.doc_count, dtype=np.float32)

        for term in query_terms:
            if term not in self.df:
                continue
            # IDF Calculation
            idf = math.log((self.doc_count - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0)
            
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lengths[i] / self.avg_doc_len))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom

        # Min-Max Normalization ke range [0, 1]
        max_score = np.max(scores)
        if max_score > 0:
            scores /= max_score
        return scores

class HybridSearch:
    def __init__(self, corpus: list[dict], model: str = EMBED_MODEL):
        self.corpus = corpus
        self.model = model
        self.bm25 = BM25Scorer([d["text"] for d in corpus])
        
        # Pre-compute semantic vectors
        texts = [f"search_document: {d['text']}" for d in corpus]
        resp = client.embeddings.create(input=texts, model=self.model)
        vecs = np.array([e.embedding for e in sorted(resp.data, key=lambda x: x.index)], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.matrix = vecs / np.where(norms == 0, 1, norms)

    def search(self, query: str, k: int = 3, alpha: float = 0.5) -> list[HybridResult]:
        # 1. Semantic Score
        q_resp = client.embeddings.create(input=[f"search_query: {query}"], model=self.model)
        q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm
        semantic_scores = self.matrix @ q_vec

        # 2. Keyword Score
        keyword_scores = self.bm25.score(query)

        # 3. Blended Score
        final_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

        top_idx = np.argsort(final_scores)[::-1][:k]
        return [
            HybridResult(
                doc_id=self.corpus[i]["id"],
                text=self.corpus[i]["text"],
                final_score=float(final_scores[i]),
                semantic_score=float(semantic_scores[i]),
                keyword_score=float(keyword_scores[i])
            )
            for i in top_idx
        ]

# --- Pengujian ---
if __name__ == "__main__":
    corpus = [
        {"id": "d1", "text": "Python tutorial for beginner developers learning algorithms."},
        {"id": "d2", "text": "Deep learning models using PyTorch framework for modern neural networks."},
        {"id": "d3", "text": "Python programming language features and system performance."},
    ]

    engine = HybridSearch(corpus)
    print("⚡ Hybrid Search Result (alpha = 0.7):")
    results = engine.search("Python deep learning", k=2, alpha=0.7)
    for r in results:
        print(f"  [{r.final_score:.4f}] (sem={r.semantic_score:.2f}, kw={r.keyword_score:.2f}) | {r.doc_id}: {r.text}")