import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]  # Minimal salah satu kata kunci harus muncul di respon
    must_be_json: bool = False

def evaluate_prompt(system: str, cases: list[EvalCase], model: str = "qwen2.5") -> dict:
    """Jalankan prompt ke seluruh test cases dan kembalikan pass rate + detail pengujian."""
    results = []
    
    for case in cases:
        response = client.chat.completions.create(
            model=model,
            max_tokens=256,
            temperature=0.0,  # Memastikan pengujian deterministik
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": case.input_text}
            ]
        )
        text = (response.choices[0].message.content or "").strip()
        
        # 1. Cek kecocokan keyword
        keyword_hit = any(kw.lower() in text.lower() for kw in case.expected_keywords)
        
        # 2. Cek kevalidan JSON jika diwajibkan
        json_valid = True
        if case.must_be_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                json_valid = False
                
        passed = keyword_hit and json_valid
        
        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response_preview": text[:80],
        })
        
    pass_rate = sum(r["passed"] for r in results) / len(results) if results else 0.0
    return {"pass_rate": pass_rate, "results": results}

# --- System Prompt & Data Uji Coba ---
CLASSIFY_SYSTEM = """Classify the AI task as one of: CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

test_cases = [
    EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
    EvalCase("Write a product description for headphones.", ["GENERATION"]),
    EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
    EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
    EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
]

# --- Pengujian ---
if __name__ == "__main__":
    print("⚡ Menjalankan evaluasi prompt otomatis di Ollama lokal...\n")
    report = evaluate_prompt(CLASSIFY_SYSTEM, test_cases)
    
    print(f"Pass rate: {report['pass_rate']:.0%}")
    for r in report["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f" [{status}] {r['input']!r} → {r['response_preview']!r}")