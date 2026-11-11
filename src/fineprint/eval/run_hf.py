"""Run a Hugging Face chat model on a GPU over an eval set, in batches, then score it (decision D35).

Same prompt (prompt.py), same few-shot examples (run_gguf.pick_shots), same scoring as the Mac
runner, so B1/B2 numbers are comparable. Differences: fp16 weights instead of the 4-bit GGUF, and
batched greedy decoding on a GPU (e.g. Kaggle T4) instead of one chunk at a time on the Mac.

Predictions are appended per batch, so a run that's interrupted resumes where it stopped.

Usage (on a GPU machine):
    python -m fineprint.eval.run_hf --system b1_qwen3b_zeroshot --set test-500
    python -m fineprint.eval.run_hf --system b2_qwen3b_fewshot --set test-500 --shots 3
    python -m fineprint.eval.run_hf --system ft_qwen3b_v1 --set test-500 --adapter /kaggle/working/ft/adapter
"""

import argparse
import json
import time

from fineprint.eval.run_gguf import pick_shots
from fineprint.eval.score import build_table, predictions_path, score_run
from fineprint.eval.sets import SETS, load_eval_set
from fineprint.prompt import build_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--set", dest="eval_set", required=True, choices=SETS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", help="LoRA adapter folder from training/train.py (merged into fp16 weights)")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, help="only the first N examples (smoke test)")
    args = ap.parse_args()

    examples = load_eval_set(args.eval_set)[: args.limit]
    path = predictions_path(args.system, args.eval_set)
    done = set()
    if path.exists():
        with open(path) as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [e for e in examples if e["id"] not in done]
    todo.sort(key=lambda e: len(e["input"]))  # similar lengths per batch → less padding
    print(f"{len(done)} done, {len(todo)} to go → {path}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
    if args.adapter:  # fine-tuned run: same fp16 base + our LoRA weights, merged for speed (D37)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    shots = pick_shots(args.shots) if args.shots else []

    def prompt(chunk: str) -> str:
        messages = build_messages(chunk)
        messages = messages[:1] + [m for pair in shots for m in pair] + messages[1:]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.perf_counter()
    with open(path, "a") as f:
        for i in range(0, len(todo), args.batch):
            part = todo[i:i + args.batch]
            enc = tok([prompt(e["input"]) for e in part], return_tensors="pt", padding=True).to("cuda")
            t = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            per = (time.perf_counter() - t) / len(part)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for e, text in zip(part, texts):
                f.write(json.dumps({"id": e["id"], "output": text, "seconds": round(per, 3)},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i + len(part)}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)

    if args.limit is None:
        notes = ((f"adapter {args.adapter}, " if args.adapter else "")
                 + f"{args.shots}-shot, fp16 on GPU, batch {args.batch} (s/example is per GPU batch slot)")
        r = score_run(args.system, args.eval_set, notes=notes)
        print(json.dumps({k: v for k, v in r.items() if k != "per_category"}, indent=2))
        print(build_table())


if __name__ == "__main__":
    main()
