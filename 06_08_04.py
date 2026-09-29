import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def stream_to_file(prompt: str, output_path: str, model: str = "qwen2.5") -> None:
    """Mengalirkan token dari API dan langsung menuliskannya ke file secara real-time."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print(f"📝 Menulis streaming response ke file: '{output_path}'...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                text_chunk = chunk.choices[0].delta.content
                # Tulis ke file dan paksakan buffer disk terbaca seketika
                f.write(text_chunk)
                f.flush()
                # Tampilkan di console untuk verifikasi
                print(text_chunk, end="", flush=True)

    print(f"\n\n Penulisan selesai. File disimpan di '{output_path}'.")

# --- Pengujian ---
if __name__ == "__main__":
    prompt_text = "Write a short 3-line poem about open-source software."
    output_file = "output_response.txt"
    
    stream_to_file(prompt_text, output_file)