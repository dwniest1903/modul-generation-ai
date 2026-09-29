from dataclasses import dataclass, field
from string import Formatter
from typing import Any

@dataclass
class PromptTemplate:
    """Template prompt yang dapat digunakan kembali dan mendukung versi (versioned)."""
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self):
        formatter = Formatter()
        combined = self.system + self.user
        self.required_vars = [
            fname for _, fname, _, _ in formatter.parse(combined)
            if fname is not None
        ]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        missing = set(self.required_vars) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.system.format(**kwargs), self.user.format(**kwargs)


# 1. Definisi Template QA
QA_TEMPLATE = PromptTemplate(
    name="question_answering",
    version="1.2",
    system="You are a {domain} expert. Answer questions accurately and concisely. Cite sources when possible. If you are unsure, say so.",
    user="Question: {question}\n\nContext:\n{context}",
)

# 2. Definisi Template Ringkasan Dokumen (Diperbaiki)
SUMMARY_TEMPLATE = PromptTemplate(
    name="document_summary",
    version="1.0",
    system="You are a technical writer. Summarise documents clearly for a {audience} audience.",
    user="Summarise the following in {max_sentences} sentences or fewer:\n\n{document}",
)

# --- Pengujian ---
if __name__ == "__main__":
    system_rendered, user_rendered = QA_TEMPLATE.render(
        domain="machine learning",
        question="What is the vanishing gradient problem?",
        context="Gradients in deep networks are computed via backpropagation...",
    )

    print("=== RESULT SYSTEM PROMPT ===")
    print(system_rendered)
    print("\n=== RESULT USER PROMPT ===")
    print(user_rendered)
    print("\n=== REQUIRED VARIABLES ===")
    print(QA_TEMPLATE.required_vars)