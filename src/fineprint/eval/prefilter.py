"""Would a DeBERTa pre-filter make documents faster without losing flags? (decision D47)

Idea: run the small DeBERTa classifier first and skip the 3B model on chunks it calls harmless. Estimated from
existing test-set predictions (no model needed): DeBERTa's answers (`b3_deberta_small`) vs the v2 model with the
0.9 threshold (`ft_qwen3b_v2_knob`), with the M1 latencies from D40 (1.5 s per empty answer, 5.4 s per flagged).

    uv run python -m fineprint.eval.prefilter
"""

import json

from fineprint.config import RESULTS, data_path, load_config
from fineprint.eval.metrics import parse_output
from fineprint.eval.score import predictions_path

THRESHOLD = 0.9
SECONDS_EMPTY, SECONDS_FLAGGED = 1.5, 5.4  # D40, M1 with the 4-bit model
REAL_DOC_FLAGGED_SHARE = 4991 / 9086  # chunks the model flagged over whole val/test services (D43)


def _jsonl(path) -> dict[str, dict]:
    rows = (json.loads(line) for line in path.read_text().splitlines())
    return {r["id"]: r for r in rows}


def main() -> None:
    cats = set(load_config("taxonomy")["categories"])
    gold = {i: json.loads(r["messages"][-1]["content"])["flags"]
            for i, r in _jsonl(data_path("processed") / "test.jsonl").items()}
    deb = _jsonl(predictions_path("b3_deberta_small", "test"))
    llm = _jsonl(predictions_path("ft_qwen3b_v2_knob", "test"))
    c = dict.fromkeys(["llm_empty", "llm_flagged", "skip_empty", "skip_flagged", "hits", "hits_lost"], 0)
    for i, g in gold.items():
        skipped = not parse_output(deb[i]["output"], cats)["flags"]
        p = llm[i]
        flags = parse_output(p["forced"], cats)["flags"] if p["p_empty"] < THRESHOLD else []
        hits = len({f["category"] for f in flags} & {f["category"] for f in g})
        kind = "flagged" if flags else "empty"
        c[f"llm_{kind}"] += 1
        c["hits"] += hits
        if skipped:
            c[f"skip_{kind}"] += 1
            c["hits_lost"] += hits
    se, sf = c["skip_empty"] / c["llm_empty"], c["skip_flagged"] / c["llm_flagged"]
    f = REAL_DOC_FLAGGED_SHARE
    full = (1 - f) * SECONDS_EMPTY + f * SECONDS_FLAGGED
    saved = (1 - f) * se * SECONDS_EMPTY + f * sf * SECONDS_FLAGGED
    report = {**c, "chunks": len(gold), "skipped_share_of_llm_empty": round(se, 3),
              "skipped_share_of_llm_flagged": round(sf, 3),
              "correct_hits_lost": round(c["hits_lost"] / c["hits"], 3),
              "time_saved_real_docs": round(saved / full, 3),
              "note": "time saved excludes DeBERTa's own run time; DeBERTa at its default cut-off (no probabilities saved)"}
    (RESULTS / "prefilter.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
