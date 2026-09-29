import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# System prompt yang kuat: peran, aturan, dan format yang jelas
STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.
Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters
Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.
Format:
Return your review as a numbered list. Each item: Issue → Impact → Fix."""

messages = [
    # Masukkan system prompt sebagai elemen pertama dengan role "system"
    {"role": "system", "content": STRONG_SYSTEM},
    {
        "role": "user", 
        "content": """Review this function:
def get_user(user_id):
    key = os.getenv('DB_KEY')
    result = requests.get(f'http://db/{user_id}?key={key}')
    return result.json()"""
    }
]

# 2. Panggil Qwen 2.5 lokal
response = client.chat.completions.create(
    model="qwen2.5",
    max_tokens=1024,
    messages=messages
)

# 3. Tampilkan hasil review
print("--- Hasil Code Review dari Qwen 2.5 ---\n")
print(response.choices[0].message.content)