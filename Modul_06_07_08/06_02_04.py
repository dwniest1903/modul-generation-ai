import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 2. Panggil API dengan parameter stream=True
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "user", "content": "List 5 use cases for vector databases."}
    ],
    max_tokens=512,
    stream=True  # Mengaktifkan efek ketik/streaming real-time
)

prompt_tokens = 0
completion_tokens = 0

print("--- Response Streaming ---")
for chunk in response:
    # Mengambil potongan teks yang dikirim secara berkala
    if chunk.choices and chunk.choices[0].delta.content:
        content = chunk.choices[0].delta.content
        print(content, end="", flush=True)

    # Mengambil token usage jika dikirimkan di potongan akhir
    if hasattr(chunk, "usage") and chunk.usage:
        prompt_tokens = chunk.usage.prompt_tokens
        completion_tokens = chunk.usage.completion_tokens

print("\n") # Baris baru setelah streaming selesai

# 3. Tampilkan total token (jika data usage tersedia)
if prompt_tokens or completion_tokens:
    print(f"Total tokens: {prompt_tokens + completion_tokens}")