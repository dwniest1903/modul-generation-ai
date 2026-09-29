import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal (menggantikan OpenAI cloud)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Key dummy wajib diisi string
)

# 2. Panggil model Qwen 2.5 lokal
response = client.chat.completions.create(
    model="qwen2.5",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "What is the difference between RAG and fine-tuning?"}
    ]
)

# 3. Tampilkan jawaban dan statistik token
print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")