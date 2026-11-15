"""Assemble the Hugging Face Space folder (decision D42): dist/space/ = app + the inference code it imports +
configs + precomputed examples. The model is not in it; the app downloads the GGUF from the model repo at start.

Usage:  uv run python app/build_space.py   then upload dist/space/ to the Space repo
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "space"
CODE = ["__init__.py", "config.py", "chunk.py", "prompt.py", "data/__init__.py", "data/clean.py", "data/stats.py",
        "eval/__init__.py", "eval/metrics.py", "inference/__init__.py", "inference/gguf.py",
        "inference/grade.py", "inference/pipeline.py"]
CONFIGS = ["data.yaml", "taxonomy.yaml", "grade.yaml"]

README = """---
title: Fine Print
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {gradio}
python_version: "3.12"
app_file: app/app.py
pinned: false
license: other
short_description: A 3B model that flags risky Terms of Service clauses
---

Fine Print flags the clauses that matter in Terms of Service, with severity, an explanation and the exact quote.
Fine Print estimate, not legal advice, and not ToS;DR's grade. Training data: [ToS;DR](https://tosdr.org)
(CC BY-SA 3.0). Base model: Qwen2.5-3B-Instruct (Qwen licence).
"""

REQUIREMENTS = """--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
llama-cpp-python==0.3.36
huggingface-hub>=0.26
pyyaml>=6.0
pandas>=2.2
langdetect>=1.0.9
"""


def main() -> None:
    import gradio

    shutil.rmtree(OUT, ignore_errors=True)
    for f in CODE:
        dst = OUT / "src" / "fineprint" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = ROOT / "src" / "fineprint" / f
        shutil.copy(src, dst) if src.exists() else dst.write_text("")
    (OUT / "configs").mkdir()
    for f in CONFIGS:
        shutil.copy(ROOT / "configs" / f, OUT / "configs" / f)
    (OUT / "app").mkdir()
    for f in ("app.py", "examples.json"):
        shutil.copy(ROOT / "app" / f, OUT / "app" / f)
    (OUT / "README.md").write_text(README.format(gradio=gradio.__version__))
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    print(OUT, sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
