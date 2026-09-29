from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

COT_RANKING_PROMPT = """You are an AI Evaluation Analyst. Given a list of evaluation scores across multiple tasks for different models:

1. Think step-by-step inside <thinking> tags: calculate the average score for each model across all tasks, then determine their final rank.
2. Write your final answer inside <answer> tags:
   - Provide a clear ranked list of the models.
   - Provide EXACTLY a 2-sentence executive recommendation based on the performance tradeoffs.

Input Data:
{input_data}"""

# 3 Input Data Berbeda
inputs = [
    """- Model A: Math=85, Coding=90, Reasoning=80
- Model B: Math=92, Coding=75, Reasoning=88
- Model C: Math=70, Coding=95, Reasoning=85""",

    """- Qwen 2.5: Math=90, Summarization=92, Translation=88
- Llama 3.1: Math=88, Summarization=90, Translation=85
- Claude Sonnet: Math=95, Summarization=96, Translation=94""",

    """- AlphaLLM: Latency=120ms, Accuracy=82%, InstructionFollow=90%
- BetaLLM: Latency=450ms, Accuracy=91%, InstructionFollow=95%
- GammaLLM: Latency=80ms, Accuracy=75%, InstructionFollow=80%"""
]

# --- Pengujian ---
if __name__ == "__main__":
    print("⚡ Menjalankan Chain-of-Thought Ranking pada 3 Varian Input Data...\n")
    for i, data in enumerate(inputs, 1):
        print(f"==================== TEST INPUT #{i} ====================")
        response = client.chat.completions.create(
            model="qwen2.5",
            temperature=0.0,
            messages=[{"role": "user", "content": COT_RANKING_PROMPT.format(input_data=data)}]
        )
        print(response.choices[0].message.content)
        print("\n")