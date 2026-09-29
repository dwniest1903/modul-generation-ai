import json
import re
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def safe_json_parse(text: str, model: str = "qwen2.5") -> dict:
    """Mencoba parsing JSON, membersihkan markdown fence, dan meminta LLM memperbaiki jika gagal."""
    # Langkah 1: Direct Parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Langkah 2: Cleaning Markdown Fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Langkah 3: Ask LLM to Repair Invalid JSON
    print("⚠️ Format JSON rusak detected! Mengirim ke Qwen 2.5 untuk auto-repair...")
    repair_prompt = f"""Fix the following invalid JSON and return ONLY the valid raw JSON object. 
Do not add markdown, code blocks, or explanations.

Invalid JSON:
{text}"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": repair_prompt}]
    )
    
    repaired_text = response.choices[0].message.content or "{}"
    return json.loads(repaired_text)

# --- Pengujian ---
if __name__ == "__main__":
    # Test case 1: Broken JSON with missing quote & trailing comma
    broken_json = '```json\n{"name": "Qwen", "age": 2, "status": "active",}\n```'
    
    result = safe_json_parse(broken_json)
    print("\n✅ Hasil Parse JSON Berhasil Diselamatkan:")
    print(json.dumps(result, indent=2))