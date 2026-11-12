"""The exact prompt and output format. Shared by dataset building, training and inference,
so all three always agree (decision D27).
"""

import json
from functools import lru_cache

from fineprint.config import load_config

SEVERITIES = ("good", "neutral", "bad", "blocker")

# Decoding setting shared by every runner (HF on Kaggle, llama.cpp on the Mac). 1.05 is Qwen2.5's built-in default,
# which `generate()` applied silently to every result up to v0.5; written down so all paths decode alike (D38).
REPETITION_PENALTY = 1.05


@lru_cache(maxsize=1)  # constant per run; re-reading the YAML per example was the builder's bottleneck
def system_prompt() -> str:
    """Short on purpose: it's repeated for every chunk, and prompt length drives CPU latency.
    The fine-tuned model learns what each category means from the examples."""
    categories = ", ".join(load_config("taxonomy")["categories"])
    return (
        "You review excerpts of Terms of Service and privacy policies. "
        "Return JSON only: {\"flags\": [...]}, one flag per notable clause, or {\"flags\": []} if none. "
        "Each flag: {\"category\", \"severity\", \"explanation\", \"quote\"}. "
        f"category is one of: {categories}. "
        f"severity is one of: {', '.join(SEVERITIES)}. "
        "explanation is one plain-English sentence for a non-lawyer. "
        "quote is copied exactly from the excerpt."
    )


def format_target(flags: list[dict]) -> str:
    """Canonical JSON answer: fixed key order, compact, non-ASCII kept readable."""
    ordered = [{k: f[k] for k in ("category", "severity", "explanation", "quote")} for f in flags]
    return json.dumps({"flags": ordered}, ensure_ascii=False)


def build_messages(chunk: str, flags: list[dict] | None = None) -> list[dict]:
    """Chat messages for one example; omit flags at inference time."""
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": chunk}]
    if flags is not None:
        messages.append({"role": "assistant", "content": format_target(flags)})
    return messages
