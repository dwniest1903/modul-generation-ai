import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
openai_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

EMBED_MODEL = "nomic-embed-text"

@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

def embed_texts(texts: list[str], input_type: str = "document") -> np.ndarray:
    """Membuat dan menormalisasi embedding menggunakan model lokal nomic-embed-text."""
    # Tambahkan prefix pencarian yang sesuai untuk nomic-embed-text
    prefix = "search_document: " if input_type == "document" else "search_query: "
    formatted_texts = [f"{prefix}{t}" for t in texts]

    resp = openai_client.embeddings.create(
        input=formatted_texts, 
        model=EMBED_MODEL
    )
    
    vecs = np.array(
        [e.embedding for e in sorted(resp.data, key=lambda x: x.index)], 
        dtype=np.float32
    )
    
    # Normalisasi ke unit length
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)

class FilteredVectorStore:
    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument]) -> None:
        """Membuat embedding dokumen dan menyimpannya ke memori."""
        embeddings = embed_texts([d.text for d in docs], input_type="document")
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
            self._docs.append(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        """Pencarian semantik dengan pre-filtering metadata opsional."""
        # 1. Terapkan pre-filter metadata
        candidates = self._docs if filter_fn is None else [d for d in self._docs if filter_fn(d)]
        if not candidates:
            return []

        # 2. Embed kueri pencarian
        q_vec = embed_texts([query], input_type="query")[0]

        # 3. Hitung skor cosine similarity (Dot product matriks x vektor query)
        matrix = np.array([d.embedding for d in candidates], dtype=np.float32)
        scores = matrix @ q_vec

        # 4. Ambil top-k indeks tertinggi
        k = min(k, len(candidates))
        top_idx = np.argsort(scores)[::-1][:k]
        
        return [(candidates[i], float(scores[i])) for i in top_idx]


# --- Pengujian & Demo ---
if __name__ == "__main__":
    docs = [
        FilteredDocument("a1", "GPT-4o supports vision and function calling.", {"category": "openai", "year": 2024}),
        FilteredDocument("a2", "Claude 3.5 Sonnet excels at coding tasks.", {"category": "anthropic", "year": 2024}),
        FilteredDocument("a3", "GPT-4o-mini is a smaller, cheaper model.", {"category": "openai", "year": 2024}),
        FilteredDocument("a4", "Claude Opus 4 is Anthropic's most capable model.", {"category": "anthropic", "year": 2025}),
        FilteredDocument("a5", "GPT-4 Turbo has a 128K context window.", {"category": "openai", "year": 2023}),
    ]

    fstore = FilteredVectorStore()
    fstore.add(docs)

    # 1. Pencarian tanpa filter (ke seluruh dokumen)
    print("=== All docs ===")
    results = fstore.search("which model is good at coding?", k=3)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.id}: {doc.text}")

    # 2. Pencarian terfilter (hanya dokumen Anthropic)
    print("\n=== Anthropic only ===")
    results = fstore.search(
        "which model is good at coding?", 
        k=3, 
        filter_fn=lambda d: d.metadata["category"] == "anthropic"
    )
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.id}: {doc.text}")