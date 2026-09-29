import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Menghitung cosine similarity antara dua vektor 1-D."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Menghitung pairwise cosine similarity untuk seluruh baris di dalam matriks.
    Mengembalikan matriks berukuran (n, n).
    """
    # 1. Normalisasi seluruh baris matriks menjadi unit vectors (panjang = 1.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = matrix / norms

    # 2. Dot product dari unit vectors sama dengan cosine similarity
    return (normed @ normed.T).astype(np.float32)

# --- Pengujian ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((4, 8)).astype(np.float32)

    sim_matrix = pairwise_similarity(vecs)

    print("=== Pairwise Cosine Similarities ===")
    for i in range(4):
        for j in range(i + 1, 4):
            print(f"  vec[{i}] vs vec[{j}] : {sim_matrix[i, j]:.4f}")

    print("\n=== Matriks Diagonal (Self-Similarity) ===")
    print(f"Diagonal: {np.diag(sim_matrix)}")