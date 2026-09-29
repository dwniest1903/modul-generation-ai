import time
import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def retry_on_rate_limit(client: OpenAI, messages: list, model: str = "qwen2.5", max_retries: int = 5):
    """Mencoba ulang pemanggilan API dengan exponential backoff jika terkena RateLimitError."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response
        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"❌ Terkena RateLimitError. Batas percobaan ({max_retries}) telah habis.")
                raise e
            
            # Perhitungan Exponential Backoff: 1s, 2s, 4s, 8s, ...
            wait_time = 2 ** attempt
            print(f"⚠️ RateLimitError terdeteksi (Percobaan {attempt + 1}/{max_retries}). Menunggu {wait_time} detik...")
            time.sleep(wait_time)

# --- Pengujian ---
if __name__ == "__main__":
    test_messages = [{"role": "user", "content": "Hello!"}]
    try:
        res = retry_on_rate_limit(client, test_messages)
        print(" Jawaban AI:", res.choices[0].message.content)
    except Exception as e:
        print("Error:", e)