"""B0b — TF-IDF + logistic regression, one classifier per category (decision D33).

The classic cheap text classifier: if it matches the fine-tuned LLM on detection, the LLM's value is
only in explanations and quotes. Trains in about a minute on the Mac.

- Input features: TF-IDF of word 1–2-grams of the chunk.
- One-vs-rest logistic regression → a probability per category.
- One decision threshold, picked on *val* to maximise category macro-F1 (test is never used to tune).
- Quote: the sentence of the chunk with the highest probability for that category.
- Severity: the category's most common severity in train; explanation: the category description.

Usage:  uv run python -m fineprint.baselines.tfidf
"""

import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from fineprint.baselines.keywords import majority_severity
from fineprint.chunk import split_sentences
from fineprint.config import load_config
from fineprint.eval.metrics import score
from fineprint.eval.score import score_run, write_predictions
from fineprint.eval.sets import _processed, load_eval_set
from fineprint.prompt import format_target

SYSTEM = "b0b_tfidf_logreg"
THRESHOLDS = np.arange(0.1, 0.75, 0.05)


def fit(train: list[dict]):
    categories = list(load_config("taxonomy")["categories"])
    mlb = MultiLabelBinarizer(classes=categories)
    y = mlb.fit_transform([{f["category"] for f in ex["flags"]} for ex in train])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300_000, sublinear_tf=True)
    x = vec.fit_transform([ex["input"] for ex in train])
    x.sort_indices()  # else joblib workers try to sort a read-only memory-mapped copy and crash
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000, solver="liblinear"), n_jobs=-1)
    clf.fit(x, y)
    return vec, clf, categories


def predict_all(examples, vec, clf, categories, threshold, severity, descriptions) -> dict[str, str]:
    probs = clf.predict_proba(vec.transform([ex["input"] for ex in examples]))
    outputs = {}
    for ex, p in zip(examples, probs):
        cats = [i for i in np.argsort(-p) if p[i] >= threshold]
        flags = []
        if cats:
            spans = split_sentences(ex["input"], 800)
            sents = [ex["input"][s:e] for s, e in spans]
            sp = clf.predict_proba(vec.transform(sents))
            for i in cats:
                c = categories[i]
                flags.append({"category": c, "severity": severity.get(c, "neutral"),
                              "explanation": descriptions[c], "quote": sents[int(sp[:, i].argmax())]})
        outputs[ex["id"]] = format_target(flags)
    return outputs


def main() -> None:
    t0 = time.perf_counter()
    vec, clf, categories = fit(_processed("train"))
    print(f"trained in {time.perf_counter() - t0:.0f}s")
    severity, descriptions = majority_severity(), load_config("taxonomy")["categories"]

    val = load_eval_set("val")
    val_f1 = {}
    best = (-1.0, None)
    for th in THRESHOLDS:
        out = predict_all(val, vec, clf, categories, th, severity, descriptions)
        f1 = score(val, out)["category_macro_f1"]
        val_f1[round(float(th), 2)] = f1
        best = max(best, (f1, float(th)))
    threshold = round(best[1], 2)
    print("val macro-F1 by threshold:", json.dumps(val_f1), "→ chosen", threshold)

    for name in ("test", "test-500", "lexglue"):
        examples = load_eval_set(name)
        t = time.perf_counter()
        outputs = predict_all(examples, vec, clf, categories, threshold, severity, descriptions)
        per = (time.perf_counter() - t) / len(examples)
        write_predictions(SYSTEM, name, outputs, {i: {"seconds": per} for i in outputs})
        r = score_run(SYSTEM, name, notes=f"threshold {threshold} tuned on val macro-F1")
        print(name, r["detection"], "macro-F1", r["category_macro_f1"])


if __name__ == "__main__":
    main()
