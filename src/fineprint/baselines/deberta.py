"""B3 — DeBERTa-v3-small multi-label classifier (decision D33). The strongest *detection* baseline:
if it beats the fine-tuned LLM at finding risky chunks, the LLM's value is explanations and quotes.

Same output recipe as B0b so the comparison is about the model, not the post-processing:
- one sigmoid output per category, trained with binary cross-entropy on train
- one threshold tuned on val for category macro-F1
- quote = the chunk sentence the model scores highest for that category
- severity = majority per category in train; explanation = category description

Runs on Apple-silicon GPU (mps), CUDA (Kaggle) or CPU.

Usage:  uv run --extra deberta python -m fineprint.baselines.deberta [--epochs 2]
"""

import argparse
import random
import time

import numpy as np

from fineprint.baselines.keywords import majority_severity
from fineprint.chunk import split_sentences
from fineprint.config import data_path, load_config
from fineprint.eval.metrics import score
from fineprint.eval.score import score_run, write_predictions
from fineprint.eval.sets import _processed, load_eval_set
from fineprint.prompt import format_target

SYSTEM = "b3_deberta_small"
MODEL = "microsoft/deberta-v3-small"
THRESHOLDS = np.arange(0.1, 0.75, 0.05)


def main() -> None:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--limit", type=int, help="train on the first N examples (smoke test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    categories = list(load_config("taxonomy")["categories"])
    index = {c: i for i, c in enumerate(categories)}

    tok = AutoTokenizer.from_pretrained(MODEL)
    # fp32 on purpose: transformers v5 loads weights in the checkpoint's dtype (fp16 for deberta-v3),
    # and AdamW in pure fp16 turns every weight into NaN after one step (eps 1e-8 underflows to 0).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(categories), problem_type="multi_label_classification",
        dtype=torch.float32).to(device)

    train = _processed("train")[: args.limit]

    def target(ex):
        y = torch.zeros(len(categories))
        for f in ex["flags"]:
            y[index[f["category"]]] = 1.0
        return y

    def batches(texts, size):
        for i in range(0, len(texts), size):
            yield tok(texts[i:i + size], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)

    @torch.no_grad()
    def probs(texts, size=64):
        model.eval()
        out = [torch.sigmoid(model(**b).logits).float().cpu().numpy() for b in batches(texts, size)]
        return np.concatenate(out) if out else np.zeros((0, len(categories)))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        for i in range(0, len(train), args.batch):
            part = train[i:i + args.batch]
            enc = tok([e["input"] for e in part], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            loss = model(**enc, labels=torch.stack([target(e) for e in part]).to(device)).loss
            if not torch.isfinite(loss):  # fail loudly instead of saving an empty model
                raise SystemExit(f"non-finite loss at epoch {epoch} step {i // args.batch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step = i // args.batch
            if step % 200 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"trained in {time.perf_counter() - t0:.0f}s on {device}")
    if args.limit:
        return  # smoke test: timing only, don't save or evaluate
    model.save_pretrained(data_path("models") / SYSTEM)
    tok.save_pretrained(data_path("models") / SYSTEM)

    severity, descriptions = majority_severity(), load_config("taxonomy")["categories"]

    def predict(examples, threshold, chunk_probs):
        outputs = {}
        for ex, p in zip(examples, chunk_probs):
            cats = [i for i in np.argsort(-p) if p[i] >= threshold]
            flags = []
            if cats:
                sents = [ex["input"][s:e] for s, e in split_sentences(ex["input"], 800)]
                sp = probs(sents)
                for i in cats:
                    c = categories[i]
                    flags.append({"category": c, "severity": severity.get(c, "neutral"),
                                  "explanation": descriptions[c], "quote": sents[int(sp[:, i].argmax())]})
            outputs[ex["id"]] = format_target(flags)
        return outputs

    val = load_eval_set("val")
    val_p = probs([e["input"] for e in val])
    # Tune on chunk-level categories only (quotes don't affect macro-F1), so skip sentence scoring.
    best = (-1.0, None)
    for th in THRESHOLDS:
        out = {e["id"]: format_target([{"category": categories[i], "severity": "neutral",
                                        "explanation": "", "quote": ""}
                                       for i in np.where(p >= th)[0]]) for e, p in zip(val, val_p)}
        best = max(best, (score(val, out)["category_macro_f1"], float(th)))
    threshold = round(best[1], 2)
    print(f"val macro-F1 {best[0]:.4f} at threshold {threshold}")

    for name in ("test", "test-500", "lexglue"):
        examples = load_eval_set(name)
        t = time.perf_counter()
        outputs = predict(examples, threshold, probs([e["input"] for e in examples]))
        per = (time.perf_counter() - t) / len(examples)
        write_predictions(SYSTEM, name, outputs, {i: {"seconds": per} for i in outputs})
        r = score_run(SYSTEM, name, notes=f"{args.epochs} epochs on {device}, threshold {threshold}")
        print(name, r["detection"], "macro-F1", r["category_macro_f1"])


if __name__ == "__main__":
    main()
