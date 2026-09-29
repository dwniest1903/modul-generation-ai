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

# 2. Definisikan skema tool (Format Standar OpenAI / Ollama)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window size and cost per 1K tokens for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model identifier, e.g. 'gpt-4o' or 'claude-sonnet-4-5'."
                    }
                },
                "required": ["model_name"]
            }
        }
    }
]

# 3. Fungsi Python lokal yang akan dipanggil
def get_model_info(model_name: str) -> dict:
    db = {
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00, "cost_output": 15.00},
        "gpt-4o": {"context_k": 128, "cost_input": 2.50, "cost_output": 10.00},
        "gemini-1.5-pro": {"context_k": 1000, "cost_input": 1.25, "cost_output": 5.00},
    }
    return db.get(model_name, {"error": f"Unknown model: {model_name}"})

# 4. Pemanggilan API pertama - Model menganalisis pertanyaan
user_prompt = "How large is the context window of claude-sonnet-4-5?"
messages = [{"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="qwen2.5",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

# 5. Cek apakah model meminta pemanggilan fungsi/tool
if tool_calls:
    # Masukkan balasan sementara dari asisten ke dalam riwayat
    messages.append(response_message)
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"Tool called: {function_name}({function_args})")
        
        # Eksekusi fungsi lokal
        if function_name == "get_model_info":
            function_response = get_model_info(**function_args)
            print(f"Tool result: {function_response}")
            
            # Kirimkan hasil fungsi kembali ke model
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            })

    # 6. Pemanggilan API kedua - Model membaca hasil fungsi & memberikan jawaban akhir
    final_response = client.chat.completions.create(
        model="qwen2.5",
        messages=messages
    )

    print("\nFinal answer:")
    print(final_response.choices[0].message.content)
else:
    print("\nAnswer without tool:")
    print(response_message.content)