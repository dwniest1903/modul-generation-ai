import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke Ollama server lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def embed_texts(texts: list[str], input_type: str = "document", model: str = "nomic-embed-text") -> tuple[np.ndarray, int]:
    """
    Mengonversi teks menjadi embedding.
    Memodelkan parameter input_type ("document" vs "query") 
    dengan menambahkan prefix khusus untuk model nomic-embed-text.
    """
    # Tambahkan prefix sesuai input_type untuk mengoptimalkan representasi vektor
    prefix = "search_document: " if input_type == "document" else "search_query: "
    formatted_texts = [f"{prefix}{text}" for text in texts]

    response = client.embeddings.create(
        input=formatted_texts,
        model=model
    )

    # Urutkan berdasarkan indeks agar posisi vektor sesuai dengan teks input
    sorted_data = sorted(response.data, key=lambda x: x.index)
    embeddings = np.array([item.embedding for item in sorted_data], dtype=np.float32)
    
    total_tokens = response.usage.prompt_tokens if response.usage else 0
    return embeddings, total_tokens

# --- Pengujian ---
if __name__ == "__main__":
    texts_to_embed = [
        "What is RAG?",
        "Explain vector databases."
    ]

    print("⚡ Membuat embeddings lokal dengan nomic-embed-text...")
    embeddings, total_tokens = embed_texts(texts_to_embed, input_type="document")

    print(f"Shape       : {embeddings.shape}")
    print(f"Token usage : {total_tokens}")