"""Reference row, not a real baseline: flag every chunk with the most common train category.

It shows what detection scores mean on our test set, where ~65% of chunks have flags: a system that
always says "risky" already gets detection recall 1.0 and F1 ~0.79 (decision D33). A real system must
beat it on precision and on the harmless-chunk flag rates, not just on F1.

Usage:  uv run python -m fineprint.baselines.trivial
"""

from collections import Counter

from fineprint.baselines.keywords import majority_severity, train_flags
from fineprint.chunk import split_sentences
from fineprint.config import load_config
from fineprint.eval.score import score_run, write_predictions
from fineprint.eval.sets import load_eval_set
from fineprint.prompt import format_target

SYSTEM = "ref_always_flag"


def main() -> None:
    category = Counter(f["category"] for f in train_flags()).most_common(1)[0][0]
    severity = majority_severity()[category]
    explanation = load_config("taxonomy")["categories"][category]
    for name in ("test", "test-500", "lexglue"):
        outputs = {}
        for ex in load_eval_set(name):
            s, e = split_sentences(ex["input"], 800)[0]
            outputs[ex["id"]] = format_target([{"category": category, "severity": severity,
                                                "explanation": explanation,
                                                "quote": ex["input"][s:e]}])
        write_predictions(SYSTEM, name, outputs)
        r = score_run(SYSTEM, name, notes=f"always flags {category}")
        print(name, r["detection"])


if __name__ == "__main__":
    main()
