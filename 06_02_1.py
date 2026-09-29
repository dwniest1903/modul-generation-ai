import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load variabel environment dari .env
load_dotenv()

# 2. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Key dummy wajib diisi string untuk library OpenAI
)

# 3. Panggil model Qwen 2.5 dengan prompt dari modul 06_02
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "user", "content": "What is retrieval-augmented generation?"}
    ]
)

# 4. Tampilkan teks respon pertama
print(response.choices[0].message.content)

# 5. Tampilkan statistik penggunaan token
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")