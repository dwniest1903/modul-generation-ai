import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 1. Tiga Tingkatan System Prompt
BASIC_SYSTEM = "You are a code review assistant. Review the code."

INTERMEDIATE_SYSTEM = """You are a Python code reviewer.
Identify bugs, security risks, and style issues in the provided code.
Provide a clear fix for each issue found."""

EXPERT_SYSTEM = """You are a Principal Software Engineer conducting production-grade code reviews.
For the provided Python snippet:
1. Identify Critical Bugs, Security Vulnerabilities, or Performance Bottlenecks.
2. Rate Severity (High/Medium/Low).
3. Provide the Refactored Production Code.
4. Explain the underlying mechanics of WHY the fix works.
Rules: Be concise, direct, and omit conversational padding."""

# 2. 5 Snippet Kode Uji Coba
CODE_SNIPPETS = [
    "def add_item(item, target_list=[]):\n    target_list.append(item)\n    return target_list",
    "import sqlite3\ndef get_user(usr):\n    conn = sqlite3.connect('db.sq3')\n    return conn.execute(f'SELECT * FROM users WHERE name={usr}').fetchall()",
    "def read_file(path):\n    f = open(path, 'r')\n    return f.read()",
    "def calc_avg(nums):\n    return sum(nums) / len(nums)",
    "import os\ndef run_cmd(cmd):\n    os.system('ping ' + cmd)"
]

# --- Pengujian ---
if __name__ == "__main__":
    prompts = [("Basic", BASIC_SYSTEM), ("Intermediate", INTERMEDIATE_SYSTEM), ("Expert", EXPERT_SYSTEM)]
    
    print("⚡ Menjalankan evaluasi 3 level system prompt pada 5 kode snippet...\n")
    for name, sys_prompt in prompts:
        print(f"==================== LEVEL: {name} ====================")
        # Uji pada snippet pertama sebagai sampel perbandingan cepat
        response = client.chat.completions.create(
            model="qwen2.5",
            temperature=0.0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Review this snippet:\n{CODE_SNIPPETS[0]}"}
            ]
        )
        print(response.choices[0].message.content)
        print("\n" + "-"*60 + "\n")