import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# System prompt yang mewajibkan format tag XML spesifik
SYSTEM = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

# 2. Panggil Qwen 2.5 lokal
response = client.chat.completions.create(
    model="qwen2.5",
    max_tokens=512,
    temperature=0.0,  # Memastikan kepatuhan format & perhitungan presisi
    messages=[
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "A RAG pipeline retrieves 5 documents, each 400 tokens. "
                "The query is 50 tokens. The model has a 4096 token limit for context. "
                "How many tokens remain for the response?"
            )
        }
    ]
)

text = response.choices[0].message.content or ""

# 3. Ekstrak bagian <thinking> dan <answer> menggunakan Regex
thinking = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

print("Reasoning:\n", thinking.group(1).strip() if thinking else "not found")
print("\nAnswer:\n", answer.group(1).strip() if answer else "not found")