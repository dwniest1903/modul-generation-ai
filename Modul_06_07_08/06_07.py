import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Dataclass untuk menampung pesan dan respon secara terstruktur
@dataclass
class ChatMessage:
    role: str  # "user" atau "assistant"
    content: str

@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

# 2. Abstract Base Class (Kontrak utama untuk semua Provider LLM)
class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        ...

# 3. Client khusus untuk Ollama Lokal (Qwen 2.5)
class OllamaClient(BaseLLMClient):
    def __init__(self, model: str = "qwen2.5"):
        self.model = model
        # Menghubungkan ke endpoint Ollama lokal
        self._client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )

        # Tangani token usage jika ada nilai None dari API lokal
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0

        return ChatResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            model=self.model,
        )

# --- Pengujian ---
if __name__ == "__main__":
    # Gunakan interface abstrak BaseLLMClient
    client: BaseLLMClient = OllamaClient(model="qwen2.5")

    msgs = [ChatMessage(role="user", content="What is a vector database?")]
    
    print("Mengirim permintaan ke Ollama lokal...")
    result = client.chat(msgs, system="Be concise.")

    print("\n--- Jawaban AI ---")
    print(result.text)
    print("\n--- Penggunaan Token ---")
    print(f"Input tokens : {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Model used   : {result.model}")