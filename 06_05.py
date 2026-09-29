import os
import base64
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL_NAME = "llava"

# --- OPSI A: URL Gambar (Mengunduh & Konversi ke Base64 dulu) ---
def describe_image_url(url: str) -> str:
    # Unduh gambar ke memory dan ubah ke base64
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        image_data = response.read()
    
    b64_data = base64.b64encode(image_data).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_data}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see in this image."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }]
    )
    return response.choices[0].message.content

# --- OPSI B: File Gambar Lokal ---
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = Path(path).suffix.lstrip(".").lower()
    
    media_type = "jpeg" if ext in ["jpg", "jpeg"] else ext
    data_url = f"data:image/{media_type};base64,{b64}"
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }]
    )
    return response.choices[0].message.content

# --- Pengujian ---
if __name__ == "__main__":
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Sunrise_over_the_sea.jpg/1280px-Sunrise_over_the_sea.jpg"
    
    print("Menganalisis gambar menggunakan model LLaVA lokal...")
    result = describe_image_url(test_url)
    
    print("\nHasil Deskripsi Gambar:")
    print(result)