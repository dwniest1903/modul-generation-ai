import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Hubungkan ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Key dummy wajib diisi string
)

# 2. Skema Tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."}
                },
                "required": ["model_name"]
            }
        }
    }
]

# 3. Fungsi lokal Python
def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o": {"context_k": 128, "cost_input": 2.50},
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00},
    }
    return db.get(model_name, {"error": "unknown model"})

# 4. Inisialisasi percakapan
messages = [{"role": "user", "content": "What is gpt-4o's context window?"}]

# 5. Panggil API pertama ke Qwen 2.5
response = client.chat.completions.create(
    model="qwen2.5",
    tools=tools,
    messages=messages
)

# 6. Cek apakah model ingin memanggil tool
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = get_model_info(**args)

    print(f"Tool dipanggil: {name}({args})")
    print(f"Hasil Tool: {result}")

    # Tambahkan pesan assistant dan hasil tool ke riwayat
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    # Panggil API kedua untuk mendapatkan respon akhir
    final = client.chat.completions.create(
        model="qwen2.5", 
        messages=messages
    )
    print("\nJawaban Akhir:")
    print(final.choices[0].message.content)
else:
    print("\nJawaban Tanpa Tool:")
    print(response.choices[0].message.content)