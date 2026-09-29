import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 2. Definisikan 3 variasi prompt
DIRECT_PROMPT = (
    "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, "
    "and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?"
)

COT_PROMPT = (
    "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, "
    "and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?\n"
    "Think through this step by step before giving the final answer."
)

ZERO_SHOT_COT = (
    "Solve this problem. Think step by step, showing each calculation.\n"
    "Finally, state: ANSWER: $X.XXXXXX\n"
    "Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens "
    "and 400 output tokens. The model costs $3.00/M input and $15.00/M output.\n"
    "What is the daily cost?"
)

prompts = [
    ("Direct", DIRECT_PROMPT),
    ("CoT", COT_PROMPT),
    ("Zero-shot CoT", ZERO_SHOT_COT)
]

# 3. Jalankan pengujian pada Qwen 2.5 lokal
for label, prompt in prompts:
    response = client.chat.completions.create(
        model="qwen2.5",
        max_tokens=512,
        temperature=0.0,  # Memastikan penalaran matematika konsisten
        messages=[{"role": "user", "content": prompt}]
    )
    
    print(f"=== {label} ===")
    # Menampilkan 300 karakter pertama dari jawaban untuk perbandingan
    print(response.choices[0].message.content[:300])
    print("\n" + "-"*50 + "\n")