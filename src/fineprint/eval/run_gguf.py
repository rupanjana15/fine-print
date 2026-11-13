"""Run a GGUF model over an eval set, then score it. B1 (zero-shot), B2 (few-shot) and later FT.

Predictions are appended one line at a time, so an interrupted run resumes where it stopped.

Usage:
    uv run python -m fineprint.eval.run_gguf --system b1_qwen3b_zeroshot --set test-500
    uv run python -m fineprint.eval.run_gguf --system b2_qwen3b_fewshot --set test-500 --shots 3
    # fine-tuned model with the empty-answer threshold (D38, D40):
    uv run python -m fineprint.eval.run_gguf --system ft_qwen3b_v2_gguf --set test-500 \
        --model data/models/fine-print-v2-q4_k_m.gguf --threshold 0.9
"""

import argparse
import json
import random
import time

from tqdm import tqdm

from fineprint.config import data_path
from fineprint.eval.score import build_table, predictions_path, score_run
from fineprint.eval.sets import SETS, _processed, load_eval_set
from fineprint.prompt import build_messages

DEFAULT_MODEL = "qwen2.5-3b-instruct-q4_k_m.gguf"


def pick_shots(k: int) -> list[list[dict]]:
    """k fixed, short train examples: one harmless, one with one flag, one with two flags, ...

    Short (300–700 chars) to keep the prompt cheap; from train so no eval data leaks in (D34).
    """
    train = [e for e in _processed("train") if 300 <= len(e["input"]) <= 700]
    rng = random.Random(0)
    rng.shuffle(train)
    wanted = [0, 1, 2] + [1] * max(0, k - 3)
    shots = []
    for n_flags in wanted[:k]:
        ex = next(e for e in train if len(e["flags"]) == n_flags and e not in shots)
        shots.append(ex)
    return [build_messages(e["input"], e["flags"])[1:] for e in shots]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--set", dest="eval_set", required=True, choices=SETS)
    ap.add_argument("--model", default=str(data_path("models") / DEFAULT_MODEL))
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--json-grammar", action="store_true", help="constrain output to valid JSON")
    ap.add_argument("--threshold", type=float, help="fine-tuned model: empty-answer threshold t (D38)")
    ap.add_argument("--threads", type=int, help="CPU threads (default: llama.cpp's choice)")
    ap.add_argument("--limit", type=int, help="only the first N examples (smoke test)")
    args = ap.parse_args()

    from fineprint.inference.gguf import GGUFModel

    examples = load_eval_set(args.eval_set)[: args.limit]
    path = predictions_path(args.system, args.eval_set)
    done = set()
    if path.exists():
        with open(path) as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [e for e in examples if e["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go → {path}")

    shots = pick_shots(args.shots) if args.shots else None
    model = GGUFModel(args.model, json_grammar=args.json_grammar, n_threads=args.threads)
    with open(path, "a") as f:
        for ex in tqdm(todo):
            t = time.perf_counter()
            if args.threshold is None:
                row = {"output": model.generate(ex["input"], shots)}
            else:
                r = model.analyse(ex["input"], threshold=args.threshold)
                row = {"output": r["output"], "p_empty": round(r["p_empty"], 6)}
            f.write(json.dumps({"id": ex["id"], **row, "seconds": round(time.perf_counter() - t, 3)},
                               ensure_ascii=False) + "\n")
            f.flush()
    model.close()

    if args.limit is None:
        notes = (f"{args.shots}-shot" + (", JSON grammar" if args.json_grammar else "")
                 + (f", empty-answer threshold {args.threshold}, {args.model.split('/')[-1]} on the Mac"
                    if args.threshold is not None else ""))
        r = score_run(args.system, args.eval_set, notes=notes)
        print(json.dumps({k: v for k, v in r.items() if k != "per_category"}, indent=2))
        print(build_table())


if __name__ == "__main__":
    main()
