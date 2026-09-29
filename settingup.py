from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "system", "content": "Kamu adalah asisten AI yang membantu penulisan kode."},
        {"role": "user", "content": "Buatkan fungsi Python untuk menghitung deret Fibonacci."}
    ]
)

# Tambahkan baris ini di paling bawah:
print(response.choices[0].message.content)