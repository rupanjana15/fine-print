"""Evaluation sets: every system is scored on exactly the same examples (decision D32).

- val / test      our ToS;DR splits (data/processed), all examples
- test-500        a fixed random 500 of test, for systems too slow to run on all of it (LLMs on a CPU)
- lexglue         LexGLUE unfair_tos test split (1,607 sentences), external; labels mapped to our
                  categories with configs/taxonomy.yaml `lexglue_labels` (D28)
- services        every chunk of every document of the graded val/test services, no gold flags; used only to
                  grade whole services (`eval/service_grades.py`, D43)

Each example: {"id", "input", "flags"} where flags are the gold flags (category, severity, ...).
LexGLUE gold flags carry only a category: it has no severities or explanations.
"""

import hashlib
import json
import random

import pandas as pd

from fineprint.config import data_path, load_config

SUBSET_SIZE = 500
SETS = ("val", "test", "test-500", "lexglue", "services")


def _processed(split: str) -> list[dict]:
    out = []
    with open(data_path("processed") / f"{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out.append({"id": r["id"], "kind": r["kind"], "input": r["messages"][1]["content"],
                        "flags": json.loads(r["messages"][2]["content"])["flags"]})
    return out


def _lexglue() -> list[dict]:
    mapping = load_config("taxonomy")["lexglue_labels"]
    df = pd.read_parquet(data_path("raw") / "lexglue" / "test.parquet")
    out = []
    for i, r in enumerate(df.itertuples()):
        cats = sorted({mapping[n] for n in r.label_names})
        out.append({"id": f"lexglue-{i:05d}", "kind": "positive" if cats else "negative",
                    "input": r.text.strip(), "flags": [{"category": c} for c in cats]})
    return out


def load_eval_set(name: str) -> list[dict]:
    if name in ("val", "test"):
        return _processed(name)
    if name == "test-500":
        test = _processed("test")
        return sorted(random.Random(0).sample(test, SUBSET_SIZE), key=lambda e: e["id"])
    if name == "lexglue":
        return _lexglue()
    if name == "services":
        with open(data_path("processed").parent / "examples" / "service_chunks.jsonl") as f:
            return [json.loads(line) for line in f]
    raise ValueError(f"unknown eval set {name!r}; choose from {SETS}")


def set_fingerprint(examples: list[dict]) -> str:
    """Short hash of inputs + gold labels, stored with results so a rebuilt dataset is noticed."""
    h = hashlib.sha256()
    for e in examples:
        h.update(json.dumps([e["id"], e["input"], e["flags"]], ensure_ascii=False).encode())
    return h.hexdigest()[:12]
