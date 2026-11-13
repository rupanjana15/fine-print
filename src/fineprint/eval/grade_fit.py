"""Fit the A–E grade rule on ToS;DR's human grades (decision D41). Mac, seconds; no model involved.

Data: comprehensively reviewed services with a ToS;DR rating A–E. For each, the approved points we kept
(points_labeled: taxonomy category + severity) are turned into flags and counted as in `inference/grade.py`.
Fit on services in the *train* split (same service hash as the dataset, D14); report on val and test services.
This measures the rule given perfect flags; the model's own flags are graded through the same rule.

Usage:  uv run python -m fineprint.eval.grade_fit   → configs/grade.yaml + results/grade_fit.json
"""

import itertools
import json
from collections import Counter, defaultdict

import pandas as pd
import yaml

from fineprint.config import CONFIGS, RESULTS, data_path, load_config
from fineprint.data.build_dataset import split_of
from fineprint.inference.grade import GRADES, counts, grade_from_score, score

W_GOOD = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
W_BLOCKER = (1.0, 2.0, 3.0, 4.0, 6.0)


def services() -> dict[int, dict]:
    out = {}
    for f in (data_path("raw") / "tosdr" / "services").glob("page_*.json"):
        for s in json.loads(f.read_text()).get("services") or []:
            if s.get("is_comprehensively_reviewed") and s.get("rating") in GRADES:
                out[s["id"]] = {"name": s["name"], "rating": s["rating"], "flags": []}
    return out


def doc_service(doc_ids: set[int]) -> dict[int, int]:
    out = {}
    for d in doc_ids:
        f = data_path("raw") / "tosdr" / "documents" / f"doc_{d}.json"
        if f.exists():
            data = json.loads(f.read_text())
            sid = data.get("parameters", data).get("service_id")
            if sid:
                out[d] = sid
    return out


def best_cuts(scores: list[float], ratings: list[str]) -> tuple[list[float], int]:
    """4 descending cut-points maximising exact accuracy. Exact, by dynamic programming: walk the distinct
    scores from high to low; each gets one grade, and grades may only get worse (A → E) as the score drops."""
    values = sorted(set(scores), reverse=True)
    hits_at = [[sum(1 for s, r in zip(scores, ratings) if s == v and r == g) for g in GRADES] for v in values]
    n, k = len(values), len(GRADES)
    dp = [[-1] * k for _ in range(n)]
    back = [[0] * k for _ in range(n)]
    for g in range(k):
        dp[0][g] = hits_at[0][g]
    for i in range(1, n):
        for g in range(k):
            prev = max(range(g + 1), key=lambda h: dp[i - 1][h])
            dp[i][g], back[i][g] = dp[i - 1][prev] + hits_at[i][g], prev
    g = max(range(k), key=lambda h: dp[n - 1][h])
    best_hits, assign = dp[n - 1][g], [0] * n
    for i in range(n - 1, -1, -1):
        assign[i] = g
        g = back[i][g]
    # cut for grade g = lowest score assigned a grade <= g (if none, the cut sits just above everything)
    cuts = []
    for g in range(k - 1):
        lows = [values[i] for i in range(n) if assign[i] <= g]
        cuts.append(min(lows) if lows else values[0] + 1)
    return cuts, best_hits


def evaluate(rows: list[dict], p: dict) -> dict:
    pred = [grade_from_score(score(r["counts"], p["w_good"], p["w_blocker"]), p["cuts"]) for r in rows]
    gold = [r["rating"] for r in rows]
    dist = [abs(GRADES.index(a) - GRADES.index(b)) for a, b in zip(pred, gold)]
    majority = Counter(gold).most_common(1)[0][0]
    return {"n": len(rows), "exact": round(sum(d == 0 for d in dist) / len(rows), 3),
            "within_one": round(sum(d <= 1 for d in dist) / len(rows), 3),
            "majority_guess_exact": round(sum(g == majority for g in gold) / len(rows), 3),
            "confusion": {g: dict(Counter(p_ for p_, g_ in zip(pred, gold) if g_ == g)) for g in GRADES}}


def main() -> None:
    cfg = load_config("data")["split"]
    svc = services()
    pts = pd.read_parquet(data_path("interim") / "points_labeled.parquet").dropna(subset=["document_id"])
    d2s = doc_service(set(pts["document_id"].astype(int)))
    seen = defaultdict(set)
    for r in pts.itertuples():
        sid = d2s.get(int(r.document_id))
        if sid in svc and r.point_id not in seen[sid]:
            seen[sid].add(r.point_id)
            svc[sid]["flags"].append({"category": r.category, "severity": r.severity})
    by_split = defaultdict(list)
    for sid, s in svc.items():
        if s["flags"]:
            by_split[split_of(f"service:{sid}", cfg["val_fraction"], cfg["test_fraction"])].append(
                {"id": sid, "rating": s["rating"], "counts": counts(s["flags"])})
    train = by_split["train"]
    print({k: len(v) for k, v in by_split.items()}, Counter(r["rating"] for r in train))

    best = None
    for wg, wb in itertools.product(W_GOOD, W_BLOCKER):
        sc = [score(r["counts"], wg, wb) for r in train]
        cuts, hits = best_cuts(sc, [r["rating"] for r in train])
        if best is None or hits > best[0]:
            best = (hits, {"w_good": wg, "w_blocker": wb, "cuts": [round(c, 6) for c in cuts]})
    params = best[1]
    report = {"params": params, **{k: evaluate(by_split[k], params) for k in ("train", "val", "test")}}
    print(json.dumps({k: (v if k == "params" else {x: v[x] for x in ("n", "exact", "within_one",
                                                                      "majority_guess_exact")})
                      for k, v in report.items()}, indent=1))
    (CONFIGS / "grade.yaml").write_text(
        "# A–E grade rule, fitted by `python -m fineprint.eval.grade_fit` on train services (D41). Do not edit by hand.\n"
        + yaml.safe_dump(params, sort_keys=False))
    (RESULTS / "grade_fit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
