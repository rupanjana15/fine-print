"""Grade whole services from the model's own flags, and check the grade rule end to end (decision D43).

The D41 rule was fitted on ToS;DR's own (human, complete) flags. On the demo's 10 single documents the model's
grade came out more generous than ToS;DR's for 8 of 10 (D42). This measures it properly: every document of every
graded val/test service goes through the model; flags are pooled per service; the grade is compared with ToS;DR's.

    uv run python -m fineprint.eval.service_grades prepare
        Mac: val/test services' documents → chunks → data/examples/service_chunks.jsonl (eval set "services")
    python -m fineprint.eval.run_knob --set services --adapter ... --threshold 0.9 --shard i/2
        Kaggle GPU (notebooks/kaggle_services.py): p_empty + forced answer per chunk
    uv run python -m fineprint.eval.service_grades analyse
        Mac: pooled flags → (a) D41 rule as fitted on human flags; (b) the same rule re-fitted on the model's flags
        of *val* services, reported on *test* services → results/service_grades.json

Uses the full-precision v2 model on GPU (test-500 macro-F1 0.658) rather than the shipped 4-bit file (0.652):
~9,000 chunks would take days on a CPU (D40), and the two agree on 487/500 flag/no-flag answers.
"""

import argparse
import itertools
import json
from collections import Counter, defaultdict

import pandas as pd

from fineprint.chunk import chunk_text
from fineprint.config import RESULTS, ROOT, data_path, load_config
from fineprint.data.build_dataset import split_of
from fineprint.data.clean import clean_text
from fineprint.eval.grade_fit import W_BLOCKER, W_GOOD, best_cuts, doc_service, evaluate
from fineprint.eval.metrics import parse_output
from fineprint.eval.score import predictions_path
from fineprint.inference.grade import GRADES, counts

OUT = ROOT / "data" / "examples" / "service_chunks.jsonl"
SYSTEM = "ft_qwen3b_v2_knob"
THRESHOLD = 0.9


def _graded_services() -> dict[int, dict]:
    sp = load_config("data")["split"]
    out = {}
    for f in (data_path("raw") / "tosdr" / "services").glob("page_*.json"):
        for s in json.loads(f.read_text()).get("services") or []:
            if s.get("is_comprehensively_reviewed") and s.get("rating") in GRADES:
                split = split_of(f"service:{s['id']}", sp["val_fraction"], sp["test_fraction"])
                if split in ("val", "test"):
                    out[s["id"]] = {"name": s["name"], "rating": s["rating"], "split": split}
    return out


def prepare() -> None:
    svc = _graded_services()
    ch = load_config("data")["chunking"]
    rows = []
    for f in sorted((data_path("raw") / "tosdr" / "documents").glob("doc_*.json")):
        d = json.loads(f.read_text())
        d = d.get("parameters", d)
        sid = d.get("service_id")
        if sid not in svc:
            continue
        text = clean_text(d.get("text") or "")
        for k, (s, e) in enumerate(chunk_text(text, ch["target_tokens"], ch["overlap_tokens"])):
            rows.append({"id": f"svc{sid}-doc{d['id']}-{k:04d}", "kind": "document", "input": text[s:e],
                         "flags": [], "service_id": sid, "doc_id": d["id"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    by = Counter(svc[r["service_id"]]["split"] for r in rows)
    print(f"{len(rows)} chunks, {len({r['service_id'] for r in rows})} services → {OUT}", dict(by))


def _predictions() -> dict[str, dict]:
    path = predictions_path(SYSTEM, "services")
    preds = {}
    for p in sorted(path.parent.glob(f"{path.stem}*.jsonl")):  # services.jsonl or services.shard*.jsonl
        for line in p.read_text().splitlines():
            r = json.loads(line)
            preds[r["id"]] = r
    return preds


def _human_counts(sids: set[int]) -> dict[int, dict]:
    """ToS;DR's own approved points for the same services, counted the same way (as in grade_fit, D41)."""
    pts = pd.read_parquet(data_path("interim") / "points_labeled.parquet").dropna(subset=["document_id"])
    d2s = doc_service(set(pts["document_id"].astype(int)))
    seen, flags = defaultdict(set), defaultdict(list)
    for r in pts.itertuples():
        sid = d2s.get(int(r.document_id))
        if sid in sids and r.point_id not in seen[sid]:
            seen[sid].add(r.point_id)
            flags[sid].append({"category": r.category, "severity": r.severity})
    return {sid: counts(f) for sid, f in flags.items()}


def analyse() -> None:
    svc = _graded_services()
    cats = set(load_config("taxonomy")["categories"])
    chunks = [json.loads(line) for line in OUT.read_text().splitlines()]
    preds = _predictions()
    missing = [c["id"] for c in chunks if c["id"] not in preds]
    assert not missing, f"{len(missing)} chunks have no prediction, e.g. {missing[:3]}"

    flags = defaultdict(list)
    flagged = 0
    for c in chunks:
        p = preds[c["id"]]
        if p["p_empty"] < THRESHOLD:
            parsed = parse_output(p["forced"], cats)["flags"]
            flagged += bool(parsed)
            flags[c["service_id"]] += [f for f in parsed if f.get("severity")]
    rows = {s: [] for s in ("val", "test")}
    for sid, s in svc.items():
        if any(c["service_id"] == sid for c in chunks):
            rows[s["split"]].append({"id": sid, "name": s["name"], "rating": s["rating"],
                                     "counts": counts(flags[sid]), "n_flags": len(flags[sid])})

    human = load_config("grade")  # D41: fitted on ToS;DR's own flags of train services
    hc = _human_counts(set(svc))
    paired = {k: [{**r, "counts": hc[r["id"]]} for r in v if r["id"] in hc] for k, v in rows.items()}
    model_paired = {k: [r for r in v if r["id"] in hc] for k, v in rows.items()}
    report = {"system": SYSTEM, "threshold": THRESHOLD, "chunks": len(chunks), "chunks_flagged": flagged,
              "services": {k: len(v) for k, v in rows.items()},
              "model_flag_counts_mean": {k: {sev: round(sum(r["counts"][sev] for r in v) / len(v), 2)
                                             for sev in ("good", "neutral", "bad", "blocker")}
                                         for k, v in rows.items()},
              "d41_rule": {"params": human, **{k: evaluate(v, human) for k, v in rows.items()}},
              # Same services, same rule: ToS;DR's own flags vs the model's (services that have approved points).
              "paired_with_human_flags": {k: {"n": len(paired[k]),
                                              "human_flags": evaluate(paired[k], human),
                                              "model_flags": evaluate(model_paired[k], human),
                                              "human_counts_mean": {sev: round(sum(r["counts"][sev] for r in paired[k])
                                                                               / len(paired[k]), 2)
                                                                    for sev in ("good", "neutral", "bad", "blocker")}}
                                          for k in rows},
              # Ceiling for fixing blockers alone: the model's flags with ToS;DR's blocker count swapped in (D45).
              "oracle_blockers": {k: evaluate([{**r, "counts": {**r["counts"], "blocker": hc[r["id"]]["blocker"]}}
                                               for r in model_paired[k]], human) for k in rows}}

    # Re-fit the same rule on the model's flags of val services, report on test services only.
    best = None
    val = rows["val"]
    for wg, wb in itertools.product(W_GOOD, W_BLOCKER):
        sc = [wg * r["counts"]["good"] - r["counts"]["bad"] - wb * r["counts"]["blocker"] for r in val]
        cuts, hits = best_cuts(sc, [r["rating"] for r in val])
        if best is None or hits > best[0]:
            best = (hits, {"w_good": wg, "w_blocker": wb, "cuts": [round(x, 6) for x in cuts]})
    refit = best[1]
    report["refit_on_model_flags"] = {"params": refit, "val (fit)": evaluate(val, refit),
                                      "test": evaluate(rows["test"], refit)}
    report["per_service"] = {k: [{x: r[x] for x in ("name", "rating", "counts", "n_flags")} for r in v]
                             for k, v in rows.items()}
    (RESULTS / "service_grades.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"mean counts": report["model_flag_counts_mean"],
                      "d41 rule": {s: {x: report["d41_rule"][s][x] for x in ("n", "exact", "within_one")}
                                   for s in ("val", "test")},
                      "same services, D41 rule (human → model flags)": {
                          s: {"n": v["n"], **{x: (v["human_flags"][x], v["model_flags"][x]) for x in ("exact", "within_one")},
                              "human mean counts": v["human_counts_mean"]}
                          for s, v in report["paired_with_human_flags"].items()},
                      "oracle blockers": {s: {x: v[x] for x in ("n", "exact", "within_one")}
                                          for s, v in report["oracle_blockers"].items()},
                      "refit params": refit,
                      "refit": {s: {x: report["refit_on_model_flags"][s][x] for x in ("n", "exact", "within_one")}
                                for s in ("val (fit)", "test")}}, indent=1))


def blockers() -> None:
    """Where gold blocker labels go on the val/test chunk sets (D45): empty chunk, other category, or which severity."""
    cats = set(load_config("taxonomy")["categories"])
    report = {}
    for split in ("val", "test"):
        gold = {}
        for line in (data_path("processed") / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            gold[r["id"]] = json.loads(r["messages"][-1]["content"])["flags"]
        preds = {}
        for line in predictions_path(SYSTEM, split).read_text().splitlines():
            r = json.loads(line)
            preds[r["id"]] = r
        fate, said, precision = Counter(), Counter(), Counter()
        for i, g in gold.items():
            p = preds[i]
            pf = parse_output(p["forced"], cats)["flags"] if p["p_empty"] < THRESHOLD else []
            for f in (f for f in g if f["severity"] == "blocker"):
                same = [x for x in pf if x["category"] == f["category"]]
                fate["chunk empty" if not pf else "other category" if not same else "same category"] += 1
                if same:
                    said[same[0]["severity"]] += 1
            for x in (x for x in pf if x.get("severity") == "blocker"):
                gs = [f["severity"] for f in g if f["category"] == x["category"]]
                precision[gs[0] if gs else "no gold label, same category"] += 1
        report[split] = {"gold_blockers": sum(fate.values()), "fate": dict(fate),
                         "severity_when_category_right": dict(said), "predicted_blockers_gold": dict(precision)}
    (RESULTS / "blocker_trace.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "analyse", "blockers"])
    {"prepare": prepare, "analyse": analyse, "blockers": blockers}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
