import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load file .env
load_dotenv()
print("Anthropic Key:", os.environ.get("ANTHROPIC_API_KEY"))
print("OpenAI Key:", os.environ.get("OPENAI_API_KEY"))

# 2. Panggil Qwen 2.5 via Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "user", "content": "Jawab singkat saja: Apakah sistem lokal Ollama sudah siap digunakan?"}
    ]
)

print("\nJawaban Qwen 2.5:")
print(response.choices[0].message.content)