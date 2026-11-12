"""Error analysis, the mechanical part (v0.5): where does a system miss, and does a reference system catch it?

Counts only, no judgement: every number here is computed from gold labels and saved predictions. The human part
(reading a sample of misses) is written up in results/error_analysis_v1.md; nothing from it becomes a label (D38).

Unit = a (chunk, gold category) pair, the same unit as category macro-F1. A miss is one of:
- empty:    the system returned no flags at all for the chunk
- partial:  it flagged the chunk, but not this category (and nothing it predicted is confused with it)
- confused: it flagged the chunk with a different category instead (counted per (gold, predicted) pair)

Usage:
    uv run python -m fineprint.eval.errors [--system ft_qwen3b_v1] [--ref b3_deberta_small] [--set test]
Writes results/error_analysis_<system>__<set>.json and a reading sample in data/interim/ (gitignored: ToS text).
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import pairwise

from fineprint.config import RESULTS, data_path, load_config
from fineprint.eval.metrics import parse_output
from fineprint.eval.score import read_predictions
from fineprint.eval.sets import load_eval_set


def predicted_categories(raw: str, categories: set[str]) -> set[str]:
    return {f["category"] for f in parse_output(raw, categories)["flags"]}


def bucket(n: int, edges: tuple[int, ...]) -> str:
    for lo, hi in pairwise(edges):
        if lo <= n < hi:
            return f"{lo}-{hi - 1}"
    return f"{edges[-1]}+"


def analyse(system: str, ref: str, eval_set: str, sample_size: int = 60) -> dict:
    categories = set(load_config("taxonomy")["categories"])
    examples = load_eval_set(eval_set)
    sys_out = {i: r["output"] for i, r in read_predictions(system, eval_set).items()}
    ref_out = {i: r["output"] for i, r in read_predictions(ref, eval_set).items()}

    kinds = Counter()                    # miss type
    by_cat = defaultdict(Counter)        # per gold category: support, hit, ref_hit, missed_but_ref_hit
    confusions = Counter()               # (gold, predicted instead)
    by_feature = defaultdict(lambda: defaultdict(Counter))  # feature -> value -> {n, hit}
    fp_cats = Counter()                  # categories predicted on chunks with no gold flags
    misses = []

    for ex in examples:
        gold = {f["category"] for f in ex["flags"]}
        pred = predicted_categories(sys_out.get(ex["id"], ""), categories)
        refp = predicted_categories(ref_out.get(ex["id"], ""), categories)
        if not gold:
            fp_cats.update(pred)
            continue
        n_gold = str(min(len(gold), 3)) + ("+" if len(gold) >= 3 else "")
        feats = {"example_kind": ex.get("kind", "?"), "gold_flags_in_chunk": n_gold,
                 "chunk_chars": bucket(len(ex["input"]), (0, 400, 800, 1200, 1600))}
        first = {}  # category -> its first gold flag (a chunk can hold two quotes of one category)
        for f in ex["flags"]:
            first.setdefault(f["category"], f)
        for c in gold:
            hit, ref_hit = c in pred, c in refp
            q = first[c]["quote"].strip()
            heading = len(q) < 60 or q.endswith((":", "?"))  # section titles / list intros, not clauses
            by_cat[c].update({"support": 1, "hit": hit, "ref_hit": ref_hit, "missed_ref_hit": (not hit) and ref_hit,
                              "heading_like": heading})
            for k, v in {**feats, "gold_severity": first[c]["severity"], "heading_like_quote": str(heading)}.items():
                by_feature[k][v].update({"n": 1, "hit": hit})
            if hit:
                continue
            extra = pred - gold
            kind = "empty" if not pred else "confused" if extra else "partial"
            kinds[kind] += 1
            for p in extra:
                confusions[(c, p)] += 1
            gold_quote = next(f["quote"] for f in ex["flags"] if f["category"] == c)
            misses.append({"id": ex["id"], "category": c, "miss": kind, "ref_caught": ref_hit,
                           "predicted": sorted(pred), "gold_quote": gold_quote, "chunk": ex["input"]})

    total = sum(v["support"] for v in by_cat.values())
    rows = {c: {"support": v["support"], "recall": round(v["hit"] / v["support"], 3),
                "ref_recall": round(v["ref_hit"] / v["support"], 3), "missed_but_ref_caught": v["missed_ref_hit"],
                "heading_like_quotes": round(v["heading_like"] / v["support"], 3)}
            for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1]["support"])}
    report = {
        "system": system, "reference": ref, "eval_set": eval_set,
        "gold_category_instances": total,
        "missed": sum(kinds.values()),
        "miss_types": dict(kinds),
        "missed_but_ref_caught": sum(v["missed_ref_hit"] for v in by_cat.values()),
        "per_category": rows,
        "recall_by_feature": {k: {v: {"n": c["n"], "recall": round(c["hit"] / c["n"], 3)} for v, c in sorted(vals.items())}
                              for k, vals in by_feature.items()},
        "top_confusions": [{"gold": g, "predicted": p, "n": n} for (g, p), n in confusions.most_common(15)],
        "false_positive_categories_on_negatives": dict(fp_cats.most_common(10)),
    }
    out = RESULTS / f"error_analysis_{system}__{eval_set}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    # Reading sample: random misses, fixed seed, so the write-up can be re-checked against the same cases.
    sample = random.Random(0).sample(misses, min(sample_size, len(misses)))
    path = data_path("interim") / f"error_sample_{system}__{eval_set}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in sample)
    print(f"{out}\n{path} ({len(sample)} misses)")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ft_qwen3b_v1")
    ap.add_argument("--ref", default="b3_deberta_small")
    ap.add_argument("--set", dest="eval_set", default="test")
    ap.add_argument("--sample", type=int, default=60)
    a = ap.parse_args()
    r = analyse(a.system, a.ref, a.eval_set, a.sample)
    print(json.dumps({k: v for k, v in r.items() if k != "per_category"}, indent=1))
    print("\nper category (support, recall, ref recall, missed but ref caught, heading-like quotes):")
    for c, v in r["per_category"].items():
        print(f"  {c:34} {v['support']:4} {v['recall']:.2f} {v['ref_recall']:.2f} {v['missed_but_ref_caught']:4}"
              f" {v['heading_like_quotes']:.2f}")


if __name__ == "__main__":
    main()
