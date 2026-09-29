import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Gunakan model embedding lokal yang tersedia di Ollama
EMBED_MODEL = "nomic-embed-text"

def embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Mengonversi daftar teks menjadi matriks vektor NumPy berukuran (n, dim)."""
    response = client.embeddings.create(input=texts, model=model)
    
    # Urutkan berdasarkan index untuk menjamin urutan hasil sesuai dengan input
    vectors = sorted(response.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)

texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]

# 2. Hasilkan Vektor Embedding
embeddings = embed(texts)

# 3. Tampilkan Informasi Vektor
print(f"Shape: {embeddings.shape}")  # Contoh: (5, 768) tergantung dimensi model
print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")