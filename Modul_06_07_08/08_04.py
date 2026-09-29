import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

EMBED_MODEL = "nomic-embed-text"

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

# ── Embedding Helper ──────────────────────────────────────────────────────────

def embed_batch(texts: list[str], input_type: str = "document", model: str = EMBED_MODEL) -> np.ndarray:
    """Membuat embedding teks via Ollama lokal. Menghasilkan matriks float32 berukuran (n, dim)."""
    # Menambahkan prefix pencarian jika menggunakan nomic-embed-text
    prefix = "search_document: " if input_type == "document" else "search_query: "
    formatted_texts = [f"{prefix}{t}" for t in texts]

    response = client.embeddings.create(input=formatted_texts, model=model)
    vectors = sorted(response.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)

# ── Simple In-Memory Vector Store ─────────────────────────────────────────────

class VectorStore:
    """In-memory vector store untuk pencarian semantik lokal."""
    
    def __init__(self, embed_model: str = EMBED_MODEL):
        self.embed_model = embed_model
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim) ter-normalisasi

    def add_documents(self, documents: list[Document]) -> None:
        """Membuat embedding dan menambahkan dokumen ke dalam indeks."""
        texts = [d.text for d in documents]
        vectors = embed_batch(texts, input_type="document", model=self.embed_model)

        # Normalisasi vektor ke unit length
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (vectors / norms).astype(np.float32)

        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)

        # Rebuild matriks penuh
        self._matrix = np.array([d.embedding for d in self._documents], dtype=np.float32)
        print(f"Indeks sekarang berisi {len(self._documents)} dokumen.")

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Mengembalikan k dokumen paling mirip untuk sebuah query."""
        if self._matrix is None or len(self._documents) == 0:
            raise RuntimeError("Belum ada dokumen yang di-indeks.")

        # Embed & normalisasi query
        q_vec = embed_batch([query], input_type="query", model=self.embed_model)[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = (q_vec / q_norm).astype(np.float32)

        # Cosine similarity via Dot Product: matrix @ vector -> shape (n,)
        scores = self._matrix @ q_vec

        # Ambil top-k indeks tertinggi
        k = min(k, len(self._documents))
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(
                document=self._documents[int(i)],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_idx)
        ]

    @property
    def size(self) -> int:
        return len(self._documents)

# ── Demo Execution ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CORPUS = [
        Document("d01", "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation to answer questions using external knowledge."),
        Document("d02", "Vector databases store high-dimensional embeddings and enable fast approximate nearest-neighbour search using algorithms like HNSW and IVF."),
        Document("d03", "Fine-tuning adapts a pre-trained language model to a specific task by continuing training on a curated dataset with task-specific examples."),
        Document("d04", "Prompt engineering involves designing and optimising input prompts to guide language models toward producing the desired output."),
        Document("d05", "LangChain is a Python framework that provides abstractions for building applications with large language models, including chains, agents, and memory."),
        Document("d06", "Cosine similarity measures the angle between two vectors and is the standard metric for comparing text embeddings in semantic search."),
        Document("d07", "RLHF (Reinforcement Learning from Human Feedback) aligns language models with human preferences by training a reward model on human rankings."),
        Document("d08", "Chunking strategies for RAG include fixed-size chunks, sentence-aware splits, and recursive character splitting with configurable overlap."),
        Document("d09", "The transformer architecture uses self-attention mechanisms to model relationships between all tokens in a sequence simultaneously."),
        Document("d10", "Agents use language models as a reasoning engine, enabling them to plan multi-step tasks, call tools, and take actions based on observations."),
    ]

    store = VectorStore()
    store.add_documents(CORPUS)

    QUERIES = [
        "How does RAG work?",
        "What algorithms do vector databases use?",
        "How do I split documents for embedding?",
    ]

    for query in QUERIES:
        print(f"\nQuery: {query!r}")
        results = store.search(query, k=3)
        for r in results:
            print(f"  [{r.rank}] score={r.score:.4f} | {r.document.text[:80]}...")