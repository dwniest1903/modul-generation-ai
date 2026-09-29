import numpy as np
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

EMBED_MODEL = "nomic-embed-text"

@dataclass
class DuplicatePair:
    doc_id_a: str
    doc_id_b: str
    similarity: float

class DuplicateDetector:
    def __init__(self, threshold: float = 0.90, model: str = EMBED_MODEL):
        self.threshold = threshold
        self.model = model

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        # Menambahkan prefix untuk nomic-embed-text
        formatted_texts = [f"search_document: {t}" for t in texts]
        resp = client.embeddings.create(input=formatted_texts, model=self.model)
        
        vecs = np.array([e.embedding for e in sorted(resp.data, key=lambda x: x.index)], dtype=np.float32)
        
        # Normalisasi unit length
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def find_duplicates(self, documents: list[dict]) -> list[DuplicatePair]:
        """
        documents: list of dicts -> [{'id': 'd1', 'text': '...'}, ...]
        """
        texts = [doc["text"] for doc in documents]
        matrix = self._embed_documents(texts)
        
        # Hitung pairwise cosine similarity
        sim_matrix = matrix @ matrix.T
        
        duplicates = []
        n = len(documents)
        
        # Ambil segitiga atas matriks (mencegah duplikasi pasangan & self-comparison)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= self.threshold:
                    duplicates.append(DuplicatePair(
                        doc_id_a=documents[i]["id"],
                        doc_id_b=documents[j]["id"],
                        similarity=score
                    ))
                    
        return sorted(duplicates, key=lambda x: x.similarity, reverse=True)

# --- Pengujian ---
if __name__ == "__main__":
    corpus = [
        {"id": "doc1", "text": "Retrieval-Augmented Generation improves LLM responses with external data."},
        {"id": "doc2", "text": "Retrieval-Augmented Generation enhances LLM answers using external knowledge."},
        {"id": "doc3", "text": "Python is a popular language for data science and AI development."},
        {"id": "doc4", "text": "Python is widely used for data science and artificial intelligence software."},
        {"id": "doc5", "text": "The Eiffel Tower is located in Paris, France."},
    ]

    detector = DuplicateDetector(threshold=0.85)
    dups = detector.find_duplicates(corpus)

    print("⚡ Pasangan Dokumen Near-Duplicate Terdeteksi:")
    for d in dups:
        print(f"  [{d.similarity:.4f}] {d.doc_id_a} <-> {d.doc_id_b}")