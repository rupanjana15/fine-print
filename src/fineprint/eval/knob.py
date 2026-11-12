"""Empty-answer threshold, step 2 of 2: tune it on val, apply it to the test sets (decision D38). Runs on the Mac.

Input: data/predictions/ft_qwen3b_v1_knob/<set>.jsonl from `eval/run_knob.py` (p_empty + forced answer per chunk).
Rule: answer = forced (the model's best non-empty answer) if p_empty < t, else `{"flags": []}`.
t is chosen on val for category macro-F1, exactly like DeBERTa's and TF-IDF's thresholds, so the comparison is fair.

Usage:  uv run python -m fineprint.eval.knob
Writes predictions + scores for system `ft_qwen3b_v1_tuned`, and results/knob_ft_qwen3b_v1.json (the val curve).
"""

import json

from fineprint.config import RESULTS, load_config
from fineprint.eval.metrics import parse_output, score
from fineprint.eval.score import build_table, read_predictions, score_run, write_predictions
from fineprint.eval.sets import load_eval_set
from fineprint.prompt import format_target

SOURCE, TARGET = "ft_qwen3b_v1_knob", "ft_qwen3b_v1_tuned"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0)
EMPTY = format_target([])


def apply(preds: dict[str, dict], t: float) -> dict[str, str]:
    return {i: (p["forced"] if p["p_empty"] < t else EMPTY) for i, p in preds.items()}


def main() -> None:
    val = load_eval_set("val")
    vp = read_predictions(SOURCE, "val")
    curve = []
    for t in THRESHOLDS:
        s = score(val, apply(vp, t))
        curve.append({"t": t, "category_macro_f1": s["category_macro_f1"], **{k: s["detection"][k] for k in
                      ("precision", "recall", "f1")}, "flag_rate_by_kind": s["flag_rate_by_kind"]})
        print(f"t={t:<6} macro-F1 {s['category_macro_f1']:.4f}  detect P {s['detection']['precision']:.3f} "
              f"R {s['detection']['recall']:.3f}  neg flagged {s['flag_rate_by_kind'].get('negative', 0):.3f}")
    best = max(curve, key=lambda r: r["category_macro_f1"])
    t = best["t"]
    print(f"best on val: t = {t}")

    # Parity check. p_empty is the raw probability; greedy decoding also applies the repetition penalty, which
    # pushes it toward flagging, so agreement at t = 0.5 is ~79-88%, not 100% (explained in D38, eval/diag_knob.py).
    tp = read_predictions(SOURCE, "test")
    greedy = read_predictions("ft_qwen3b_v1", "test")
    cats = set(load_config("taxonomy")["categories"])
    same = sum((tp[i]["p_empty"] < 0.5) == bool(parse_output(g["output"], cats)["flags"])
               for i, g in greedy.items() if i in tp) / len(greedy)
    print(f"parity: raw t=0.5 matches penalised greedy decoding on {same:.1%} of test chunks (not 100%, see D38)")

    notes = f"empty-answer threshold t={t} tuned on val (D38)"
    for name, src in (("test", "test"), ("test-500", "test"), ("lexglue", "lexglue")):
        ids = {e["id"] for e in load_eval_set(name)}
        preds = {i: p for i, p in read_predictions(SOURCE, src).items() if i in ids}
        write_predictions(TARGET, name, apply(preds, t), {i: {"seconds": p["seconds"]} for i, p in preds.items()})
        r = score_run(TARGET, name, notes=notes)
        print(f"{name:9} macro-F1 {r['category_macro_f1']:.4f}  detect {r['detection']}")
    (RESULTS / "knob_ft_qwen3b_v1.json").write_text(json.dumps(
        {"chosen_t": t, "parity_with_greedy": round(same, 4), "val_curve": curve}, indent=2) + "\n")
    build_table()


if __name__ == "__main__":
    main()
