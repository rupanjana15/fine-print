"""Empty-answer threshold for the fine-tuned model, step 1 of 2: the GPU part (decision D38).

The fine-tuned model's answer always starts `{"flags":` and the next token decides everything: ` []` (no flags)
or ` [{"` (at least one flag). Greedy decoding picks whichever is more likely: an implicit threshold, never tuned.

For every chunk this script saves:
- p_empty: P(` []`) / (P(` []`) + P(` [{"`)) at that decision point (one forward pass)
- forced:  the greedy answer *given* the model starts a flag (prefix `{"flags": [{"` forced, then greedy)

With both saved, any threshold t can be applied later on the Mac without a GPU: answer = forced if p_empty < t,
else `{"flags": []}`. Step 2 (`eval/knob.py`) tunes t on val.

p_empty is the model's raw probability. Plain `generate()` also applies the repetition penalty (1.05, Qwen's
default, now explicit as prompt.REPETITION_PENALTY), which lowers ` []` because that token appears in the
prompt. So t = 0.5 here is *not* the same as greedy decoding: greedy flags ~11 points more often (measured by
`eval/diag_knob.py`, D38). The forced continuation below uses the same penalty as greedy decoding.

Usage (GPU):
    python -m fineprint.eval.run_knob --adapter /kaggle/working/ft/adapter --set val
Mac smoke test:
    uv run --extra train python -m fineprint.eval.run_knob --model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter <smoke adapter> --set val --limit 4 --device cpu --max-new-tokens 40
"""

import argparse
import json
import time

from fineprint.eval.score import predictions_path
from fineprint.eval.sets import SETS, load_eval_set
from fineprint.prompt import REPETITION_PENALTY, build_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PREFIX = '{"flags":'          # tokens shared by every answer
EMPTY, FLAG = " []", ' [{"'   # the two possible next tokens (checked to be single tokens below)
SYSTEM = "ft_qwen3b_v1_knob"


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="eval_set", required=True, choices=SETS)
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th example from i (one per GPU)")
    ap.add_argument("--threshold", type=float,
                    help="skip the forced answer where p_empty >= t (it would be answered empty anyway); "
                         "saves most of the time on whole documents. Leave unset to keep every forced answer (D38)")
    args = ap.parse_args()
    shard_i, shard_n = map(int, args.shard.split("/"))

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    prefix = tok(PREFIX, add_special_tokens=False)["input_ids"]
    empty_ids, flag_ids = (tok(t, add_special_tokens=False)["input_ids"] for t in (EMPTY, FLAG))
    if len(empty_ids) != 1 or len(flag_ids) != 1 or tok(PREFIX + FLAG, add_special_tokens=False)["input_ids"] != prefix + flag_ids:
        raise SystemExit("tokenizer splits the answer differently than expected; the decision point is not one token")
    e_id, f_id = empty_ids[0], flag_ids[0]

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=args.device)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    examples = load_eval_set(args.eval_set)[: args.limit][shard_i::shard_n]
    path = predictions_path(args.system, args.eval_set)
    if shard_n > 1:
        path = path.with_name(f"{path.stem}.shard{shard_i}.jsonl")
    done = set()
    if path.exists():
        with open(path) as f:
            done = {json.loads(line)["id"] for line in f}
    todo = sorted((e for e in examples if e["id"] not in done), key=lambda e: len(e["input"]))
    print(f"{len(done)} done, {len(todo)} to go → {path}", flush=True)

    def prompt_ids(chunk: str) -> list[int]:
        text = tok.apply_chat_template(build_messages(chunk), tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    def batch(seqs: list[list[int]]):
        return tok.pad({"input_ids": seqs}, return_tensors="pt").to(args.device)

    t0 = time.perf_counter()
    with open(path, "a") as f, torch.no_grad():
        for i in range(0, len(todo), args.batch):
            part = todo[i:i + args.batch]
            base = [prompt_ids(e["input"]) for e in part]
            t = time.perf_counter()
            # 1) P(empty) at the decision point: next-token distribution after `{"flags":`
            logits = model(**batch([b + prefix for b in base])).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            pe, pf = probs[:, e_id], probs[:, f_id]
            p_empty = (pe / (pe + pf)).tolist()
            # 2) the best non-empty answer: force `{"flags": [{"` and continue greedily
            need = [k for k, p in enumerate(p_empty) if args.threshold is None or p < args.threshold]
            cont = [None] * len(part)
            if need:
                enc = batch([base[k] + prefix + [f_id] for k in need])
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     repetition_penalty=REPETITION_PENALTY, pad_token_id=tok.pad_token_id)
                for k, c in zip(need, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    cont[k] = PREFIX + FLAG + c
            per = (time.perf_counter() - t) / len(part)
            f.writelines(json.dumps({"id": e["id"], "p_empty": round(p, 6), "p_two_way": round(float(a + b), 6),
                                     "forced": c, "seconds": round(per, 3)}, ensure_ascii=False) + "\n"
                         for e, p, a, b, c in zip(part, p_empty, pe.tolist(), pf.tolist(), cont))
            f.flush()
            print(f"{i + len(part)}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
