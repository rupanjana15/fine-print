"""Whole document → flags + grade (ARCHITECTURE §6, decisions D25, D40, D41).

clean → chunk (the same chunker as the dataset) → model per chunk (threshold, D38) → validate (JSON, taxonomy,
quote found in the chunk) → aggregate (one flag per issue across overlapping chunks) → grade A–E.

Usage (Mac):
    uv run --extra eval python -m fineprint.inference.pipeline --model data/models/fine-print-v2-q4_k_m.gguf \
        --file some_terms.txt
"""

import argparse
import json
import re
import time
from collections.abc import Callable

from fineprint.chunk import chunk_text
from fineprint.config import load_config
from fineprint.data.clean import clean_text
from fineprint.eval.metrics import parse_output
from fineprint.inference.grade import grade

Predictor = Callable[[str], dict]  # chunk -> {"output": raw model text, "p_empty": float}


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Character span of `quote` in `text`, ignoring differences in whitespace; None if it isn't there."""
    words = quote.split()
    if not words:
        return None
    m = re.search(r"\s+".join(map(re.escape, words)), text)
    return (m.start(), m.end()) if m else None


def aggregate(flags: list[dict]) -> list[dict]:
    """Merge flags of the same category whose quotes overlap (the same clause seen by two overlapping chunks).
    Keeps the first; flags without a located quote are kept unless an identical one already exists."""
    def same(f: dict, g: dict) -> bool:
        if f["category"] != g["category"]:
            return False
        if f["span"] and g["span"]:
            return f["span"][0] < g["span"][1] and g["span"][0] < f["span"][1]
        return f["quote"] == g["quote"]

    out: list[dict] = []
    for f in flags:
        if not any(same(f, g) for g in out):
            out.append(f)
    return out


def analyse_document(text: str, predict: Predictor) -> dict:
    t0 = time.perf_counter()
    cfg = load_config("data")["chunking"]
    cats = set(load_config("taxonomy")["categories"])
    text = clean_text(text)
    spans = chunk_text(text, cfg["target_tokens"], cfg["overlap_tokens"])
    flags, chunk_info = [], []
    for i, (s, e) in enumerate(spans):
        t = time.perf_counter()
        r = predict(text[s:e])
        parsed = parse_output(r["output"], cats)
        for f in parsed["flags"]:
            loc = locate(f["quote"], text[s:e])
            flags.append({**f, "chunk": i, "span": (s + loc[0], s + loc[1]) if loc else None,
                          "quote_found": loc is not None})
        chunk_info.append({"span": (s, e), "p_empty": r.get("p_empty"), "n_flags": len(parsed["flags"]),
                           "valid_json": parsed["strict"], "seconds": round(time.perf_counter() - t, 3)})
    merged = aggregate(flags)
    graded = grade([f for f in merged if f["severity"]])
    return {**graded, "flags": merged, "chunks": chunk_info, "n_chunks": len(spans), "chars": len(text),
            "seconds": round(time.perf_counter() - t0, 2),
            "note": "Fine Print estimate, not ToS;DR's grade and not legal advice (D20)"}


def main() -> None:
    from fineprint.inference.gguf import GGUFModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True, help="plain-text Terms of Service / privacy policy")
    ap.add_argument("--out", help="write the full result as JSON here")
    args = ap.parse_args()
    with open(args.file) as fh:
        text = fh.read()
    model = GGUFModel(args.model)
    res = analyse_document(text, model.analyse)
    model.close()
    print(f"grade {res['grade']} (score {res['score']}, {res['counts']}), {len(res['flags'])} flags, "
          f"{res['n_chunks']} chunks, {res['seconds']} s")
    for f in res["flags"]:
        print(f"- [{f['severity']}] {f['category']}: {f['explanation']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
