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

SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]

def extract_company_info(text: str) -> dict:
    response = client.chat.completions.create(
        model="qwen2.5",
        max_tokens=256,
        temperature=0.0,  # Memastikan konsistensi ekstraksi JSON
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}
        ],
        # Aktifkan format JSON mode jika didukung oleh Ollama/Qwen
        response_format={"type": "json_object"}
    )
    
    raw = response.choices[0].message.content or ""
    
    # Bersihkan markdown code fences jika model masih secara tidak sengaja menyertakannya
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]  # Hapus baris pertama (```json)
    if raw.endswith("```"):
        raw = raw.rsplit("\n", 1)[0]   # Hapus baris terakhir (```)
    raw = raw.strip()

    return json.loads(raw)

# --- Pengujian ---
if __name__ == "__main__":
    for text in texts:
        info = extract_company_info(text)
        print(json.dumps(info, indent=2))
        print()