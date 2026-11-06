"""Turn raw model outputs into scores. One function for every system (decision D32).

Metrics (ARCHITECTURE §5):
- json_valid        output is exactly one JSON object {"flags": [...]} (what the app needs)
- json_recovered    a {"flags": [...]} object could be found somewhere in the output (lenient parse,
                    used for all other metrics so a stray "```json" fence doesn't zero a system)
- detection         chunk level: gold has >= 1 flag vs prediction has >= 1 valid flag
- category          chunk level, per category: is the category among the gold / predicted ones?
                    macro-F1 = mean over categories that occur in the gold labels
- severity          for gold flags whose category was predicted: did a predicted flag of that
                    category have the same severity? (+ confusion matrix)
- quote_faithfulness  share of predicted flags whose quote appears in the input (whitespace-normalised)
- false_positive_rate by example kind (negative vs hard_negative): does it cry wolf on look-alikes?
"""

import json
import re
from collections import Counter, defaultdict

from fineprint.config import load_config
from fineprint.prompt import SEVERITIES

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def parse_output(raw: str, categories: set[str]) -> dict:
    """Return {"strict": bool, "recovered": bool, "flags": [...], "invalid_flags": int}."""
    strict, obj = False, None
    try:
        obj = json.loads(raw.strip())
        strict = isinstance(obj, dict) and isinstance(obj.get("flags"), list)
    except (json.JSONDecodeError, TypeError):
        pass
    if not strict:
        obj = None
        dec = json.JSONDecoder()
        for m in re.finditer(r"\{", raw or ""):
            try:
                cand, _ = dec.raw_decode(raw[m.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(cand, dict) and isinstance(cand.get("flags"), list):
                obj = cand
                break
    flags, invalid = [], 0
    for f in (obj or {}).get("flags", []):
        if isinstance(f, dict) and f.get("category") in categories:
            sev = f.get("severity")
            flags.append({"category": f["category"],
                          "severity": sev if sev in SEVERITIES else None,
                          "explanation": str(f.get("explanation") or ""),
                          "quote": str(f.get("quote") or "")})
        else:
            invalid += 1
    return {"strict": strict, "recovered": obj is not None, "flags": flags, "invalid_flags": invalid}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def score(examples: list[dict], outputs: dict[str, str], label_space: set[str] | None = None) -> dict:
    """examples: from eval.sets; outputs: id -> raw model output.

    label_space limits which categories count (LexGLUE only has labels for some of ours);
    predicted flags outside it are ignored.
    """
    categories = set(load_config("taxonomy")["categories"])
    space = label_space or categories
    det = Counter()
    cat = defaultdict(Counter)
    sev_conf = defaultdict(Counter)
    kinds = defaultdict(Counter)
    n_strict = n_recovered = n_invalid = n_pred_flags = n_faithful = 0
    missing = 0

    for ex in examples:
        raw = outputs.get(ex["id"])
        if raw is None:
            missing += 1
            raw = ""
        parsed = parse_output(raw, categories)
        n_strict += parsed["strict"]
        n_recovered += parsed["recovered"]
        n_invalid += parsed["invalid_flags"]
        pred = [f for f in parsed["flags"] if f["category"] in space]
        gold = [f for f in ex["flags"] if f["category"] in space]

        for f in pred:
            n_pred_flags += 1
            n_faithful += bool(f["quote"]) and _norm(f["quote"]) in _norm(ex["input"])

        g, p = bool(gold), bool(pred)
        det["tp"] += g and p
        det["fp"] += p and not g
        det["fn"] += g and not p
        kinds[ex.get("kind", "?")]["n"] += 1
        kinds[ex.get("kind", "?")]["flagged"] += p

        gc, pc = {f["category"] for f in gold}, {f["category"] for f in pred}
        for c in gc | pc:
            cat[c]["tp" if c in gc and c in pc else "fn" if c in gc else "fp"] += 1

        for gf in gold:
            if gf.get("severity") and gf["category"] in pc:
                sevs = [f["severity"] for f in pred if f["category"] == gf["category"]]
                got = gf["severity"] if gf["severity"] in sevs else sevs[0]
                sev_conf[gf["severity"]][got or "missing"] += 1

    per_cat = {c: _prf(cat[c]["tp"], cat[c]["fp"], cat[c]["fn"]) for c in sorted(space) if cat[c]}
    gold_cats = [c for c in per_cat if per_cat[c]["tp"] + per_cat[c]["fn"] > 0]
    micro = _prf(*(sum(cat[c][k] for c in space) for k in ("tp", "fp", "fn")))
    sev_total = sum(sum(v.values()) for v in sev_conf.values())
    sev_right = sum(sev_conf[s][s] for s in sev_conf)
    n = len(examples)
    return {
        "n_examples": n,
        "missing_outputs": missing,
        "json_valid": round(n_strict / n, 4),
        "json_recovered": round(n_recovered / n, 4),
        "invalid_flags": n_invalid,
        "detection": _prf(det["tp"], det["fp"], det["fn"]),
        "category_macro_f1": round(sum(per_cat[c]["f1"] for c in gold_cats) / len(gold_cats), 4)
        if gold_cats else 0.0,
        "category_micro": micro,
        "severity_accuracy": round(sev_right / sev_total, 4) if sev_total else None,
        "severity_confusion": {s: dict(sev_conf[s]) for s in SEVERITIES if s in sev_conf},
        "quote_faithfulness": round(n_faithful / n_pred_flags, 4) if n_pred_flags else None,
        "predicted_flags": n_pred_flags,
        "flag_rate_by_kind": {k: round(v["flagged"] / v["n"], 4) for k, v in sorted(kinds.items())},
        "per_category": per_cat,
    }
