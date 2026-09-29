import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Key dummy wajib diisi string
)

# 2. Panggil API dengan parameter stream=True
stream = client.chat.completions.create(
    model="qwen2.5",
    max_tokens=512,
    stream=True,
    messages=[
        {"role": "user", "content": "Explain embeddings in 3 bullet points."}
    ]
)

# 3. Iterasi setiap potongan teks (chunk) saat diterima secara real-time
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)

print()  # Baris baru setelah streaming selesai