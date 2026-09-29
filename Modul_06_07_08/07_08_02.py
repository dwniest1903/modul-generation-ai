import json
from dataclasses import dataclass, field, asdict
from string import Formatter
from typing import Any

@dataclass
class PromptTemplate:
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


class PromptLibrary:
    def __init__(self):
        self.templates: dict[str, PromptTemplate] = {}
        self.last_eval_runs: dict[str, str] = {}  # Tracks {template_name: last_evaluated_version}

    def add(self, template: PromptTemplate):
        self.templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found.")
        return self.templates[name]

    def record_eval_run(self, name: str):
        if name in self.templates:
            self.last_eval_runs[name] = self.templates[name].version

    def save_to_json(self, filepath: str):
        data = {
            "templates": {k: asdict(v) for k, v in self.templates.items()},
            "last_eval_runs": self.last_eval_runs
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Library berhasil disimpan ke {filepath}")

    def load_from_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.templates = {
            k: PromptTemplate(**v) for k, v in data.get("templates", {}).items()
        }
        self.last_eval_runs = data.get("last_eval_runs", {})
        print(f"✅ Library berhasil dimuat dari {filepath}")

# --- Pengujian ---
if __name__ == "__main__":
    lib = PromptLibrary()
    
    # Tambah template
    t1 = PromptTemplate("classifier", "Classify as {category}", "Text: {text}", version="1.1")
    lib.add(t1)
    lib.record_eval_run("classifier")

    # Save & Load
    lib.save_to_json("prompt_library.json")
    
    new_lib = PromptLibrary()
    new_lib.load_from_json("prompt_library.json")
    print("Versi Terakhir Evaluasi:", new_lib.last_eval_runs)