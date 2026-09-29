import hashlib
import sqlite3
import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
DB_PATH = "embeddings_cache.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                hash_key TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)

def _get_hash_key(text: str, model: str) -> str:
    raw = f"{model}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def embed_with_cache(text: str, model: str = "nomic-embed-text") -> np.ndarray:
    init_db()
    hash_key = _get_hash_key(text, model)

    # 1. Cek dari SQLite Cache
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM cache WHERE hash_key = ?", (hash_key,))
        row = cursor.fetchone()
        
        if row:
            print("⚡ Cache HIT! Mengambil dari database SQLite...")
            return np.frombuffer(row[0], dtype=np.float32)

    # 2. Jika Miss, Panggil Ollama API
    print("🌐 Cache MISS! Memanggil Ollama API...")
    prefix_text = f"search_document: {text}"
    resp = client.embeddings.create(input=[prefix_text], model=model)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)

    # 3. Simpan ke Cache
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO cache (hash_key, embedding) VALUES (?, ?)",
            (hash_key, vec.tobytes())
        )

    return vec

# --- Pengujian ---
if __name__ == "__main__":
    sample_text = "This is a test sentence for embedding cache."

    print("--- Panggilan Pertama ---")
    v1 = embed_with_cache(sample_text)

    print("\n--- Panggilan Kedua (Teks Sama) ---")
    v2 = embed_with_cache(sample_text)

    print(f"\nApakah kedua vektor identik? {np.array_equal(v1, v2)}")