import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
EMBED_MODEL = "nomic-embed-text"

@dataclass
class Document:
    id: str
    text: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

def embed_text(text: str, input_type: str = "document") -> np.ndarray:
    prefix = "search_document: " if input_type == "document" else "search_query: "
    resp = client.embeddings.create(input=[f"{prefix}{text}"], model=EMBED_MODEL)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

class DynamicVectorStore:
    def __init__(self):
        self._documents: list[Document] = []
        self._id_to_index: dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None

    def _rebuild_matrix(self):
        """Membangun ulang matriks dan pemetaan indeks."""
        if not self._documents:
            self._matrix = None
            self._id_to_index = {}
            return
        
        self._matrix = np.array([d.embedding for d in self._documents], dtype=np.float32)
        self._id_to_index = {d.id: idx for idx, d in enumerate(self._documents)}

    def add_documents(self, documents: list[Document]):
        for doc in documents:
            doc.embedding = embed_text(doc.text, "document")
            self._documents.append(doc)
        self._rebuild_matrix()

    def delete(self, doc_id: str) -> bool:
        """Menghapus dokumen berdasarkan doc_id dan menyinkronkan matriks."""
        if doc_id not in self._id_to_index:
            return False
        
        idx = self._id_to_index[doc_id]
        self._documents.pop(idx)
        self._rebuild_matrix()
        print(f"✅ Document '{doc_id}' deleted. Current index size: {len(self._documents)}")
        return True

    def update(self, doc_id: str, new_text: str) -> bool:
        """Memperbarui teks dokumen dan menghitung ulang embedding-nya."""
        if doc_id not in self._id_to_index:
            return False
        
        idx = self._id_to_index[doc_id]
        self._documents[idx].text = new_text
        self._documents[idx].embedding = embed_text(new_text, "document")
        self._rebuild_matrix()
        print(f"✅ Document '{doc_id}' updated.")
        return True

    @property
    def size(self) -> int:
        return len(self._documents)

# --- Pengujian ---
if __name__ == "__main__":
    store = DynamicVectorStore()
    store.add_documents([
        Document("d1", "Vector databases store embeddings."),
        Document("d2", "RAG integrates search with LLMs.")
    ])

    print(f"Awal: Size = {store.size}")
    store.update("d1", "Vector databases enable high-dimensional fast similarity search.")
    store.delete("d2")
    print(f"Akhir: Size = {store.size}")