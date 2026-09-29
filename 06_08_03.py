import asyncio
import time
import pandas as pd
from openai import AsyncOpenAI

# Inisialisasi AsyncClient untuk Ollama lokal
async_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

async def fetch_model_response(prompt: str, model: str) -> dict:
    """Mengukur latency, token usage, dan mengambil jawaban dari satu model."""
    start_time = time.perf_counter()
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        return {
            "model": model,
            "response_text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "latency_ms": latency_ms
        }
    except Exception as e:
        return {
            "model": model,
            "response_text": f"Error: {str(e)}",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0
        }

async def compare_models(prompt: str, models: list[str]) -> pd.DataFrame:
    """Memanggil prompt ke beberapa model secara async dan mengembalikan pandas DataFrame."""
    tasks = [fetch_model_response(prompt, m) for m in models]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)

# --- Pengujian ---
if __name__ == "__main__":
    # Model yang ada di Ollama Anda (contoh: qwen2.5, llava)
    test_models = ["qwen2.5", "llava"]
    test_prompt = "What is a vector database? Explain in 1 sentence."

    print("⚡ Menjalankan pembandingan model secara paralel...")
    df = asyncio.run(compare_models(test_prompt, test_models))
    
    # Tampilkan tabel DataFrame
    print("\n", df.to_string(index=False))