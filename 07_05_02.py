import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 2. Panggil Qwen 2.5 dengan pemaksaan format JSON
response = client.chat.completions.create(
    model="qwen2.5",
    response_format={"type": "json_object"},  # Mengunci output agar selalu berupa JSON yang valid
    temperature=0.0,
    messages=[
        {
            "role": "system",
            "content": (
                "Extract entities. Return JSON with this schema:\n"
                '{"people": [string], "organizations": [string], "locations": [string]}'
            )
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
        }
    ]
)

# 3. Parse dan tampilkan hasil sebagai objek dictionary Python
result = json.loads(response.choices[0].message.content or "{}")
print("--- Hasil Ekstraksi Entitas ---")
print(result)