import os
from dotenv import load_dotenv

load_dotenv()

# 1. Tabel Biaya Per 1M Token (USD)
PRICING = {
    "qwen2.5-local": {"input": 0.00, "output": 0.00},  # Gratis karena dijalankan di GPU/CPU lokal
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# 2. Batas Context Window (Jumlah token maksimum)
CONTEXT_LIMITS = {
    "qwen2.5-local": 32_768,       # Default context window Qwen 2.5
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-1.5-pro": 1_000_000,
}

# 3. Fungsi Estimasi Biaya
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Mengembalikan estimasi biaya dalam USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

# 4. Fungsi Cek Kapasitas Context Window
def fits_in_context(model: str, token_count: int, reserve_for_output: int = 2048) -> bool:
    """Mengecek apakah prompt muat dalam context window model."""
    limit = CONTEXT_LIMITS.get(model, 32_768)
    return token_count + reserve_for_output <= limit

# --- Pengujian ---
if __name__ == "__main__":
    # Pengujian Estimasi Biaya Lokal vs Cloud
    input_tokens = 500
    output_tokens = 300
    
    cost_local = estimate_cost("qwen2.5-local", input_tokens, output_tokens)
    cost_claude = estimate_cost("claude-sonnet-4-5", input_tokens, output_tokens)
    
    print(f"Estimasi biaya Qwen 2.5 Lokal : ${cost_local:.6f}")
    print(f"Estimasi biaya Claude Sonnet  : ${cost_claude:.6f}\n")
    
    # Pengujian Cek Context Window
    test_token_count = 30_000
    model_name = "qwen2.5-local"
    
    if fits_in_context(model_name, test_token_count):
        print(f"✅ Prompt ({test_token_count} token) MUAT di context window {model_name}.")
    else:
        print(f"❌ Prompt ({test_token_count} token) MELEBIHI kapasitas {model_name}!")