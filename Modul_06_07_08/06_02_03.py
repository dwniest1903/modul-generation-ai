import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def chat(system_prompt: str) -> None:
    """Fungsi chat interaktif multi-turn sederhana."""
    # Simpan system prompt di awal riwayat pesan
    history = [
        {"role": "system", "content": system_prompt}
    ]
    
    print("--- Obrolan Dimulai (Ketik 'exit' atau 'quit' untuk keluar) ---\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ("exit", "quit"):
            print("Chat selesai.")
            break
            
        # Tambahkan pesan user ke riwayat
        history.append({"role": "user", "content": user_input})
        
        # Kirim seluruh riwayat ke Qwen 2.5 lokal
        response = client.chat.completions.create(
            model="qwen2.5",
            messages=history,
            max_tokens=1024
        )
        
        assistant_text = response.choices[0].message.content
        
        # Tambahkan balasan AI ke riwayat agar kontekstual
        history.append({"role": "assistant", "content": assistant_text})
        
        print(f"\nQwen: {assistant_text}\n")

# Jalankan fungsi chat dengan tutor Python
if __name__ == "__main__":
    chat(system_prompt="You are a helpful Python tutor. Answer concisely.")