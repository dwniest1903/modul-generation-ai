import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class BudgetExceeded(Exception):
    """Exception yang dilempar saat akumulasi token melebihi batas anggaran."""
    pass

class TokenBudgetManager:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.total_tokens_used = 0

    def add_usage(self, tokens_used: int):
        """Menambahkan penggunaan token dan mengecek batas anggaran."""
        if self.total_tokens_used + tokens_used > self.max_tokens:
            excess = (self.total_tokens_used + tokens_used) - self.max_tokens
            raise BudgetExceeded(
                f"Anggaran token terlampaui! Maksimal: {self.max_tokens}, "
                f"Terpakai: {self.total_tokens_used}, Mencoba menambah: {tokens_used} (Kelebihan {excess} token)."
            )
        self.total_tokens_used += tokens_used
        print(f"📊 Token digunakan: +{tokens_used} | Total akumulasi: {self.total_tokens_used}/{self.max_tokens}")

# --- Pengujian ---
if __name__ == "__main__":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    budget = TokenBudgetManager(max_tokens=200) # Batas kecil untuk pengujian

    messages = [{"role": "user", "content": "Explain gravity in one brief sentence."}]

    try:
        for i in range(5):
            print(f"\n--- Pemanggilan API ke-{i+1} ---")
            resp = client.chat.completions.create(model="qwen2.5", messages=messages)
            
            used = resp.usage.total_tokens if resp.usage else 0
            budget.add_usage(used)
            print("Jawaban:", resp.choices[0].message.content)

    except BudgetExceeded as e:
        print(f"\n❌ {e}")