"""Why does the empty-answer threshold at t = 0.5 not reproduce greedy decoding? (diagnostic for D38, GPU)

The knob run (`run_knob.py`) and the original greedy eval (`run_hf.py`) should make the same empty / non-empty
decision at t = 0.5, but agree on only 79% of test chunks. This script measures P(empty) at the decision point
four ways on the same chunks (the first --n test chunks in the same length order and batches as both runs):

- gen_b8 / gen_b1: greedy `generate` (as run_hf), batch 8 / 1; P(empty) read from the scores at the decision step
- fwd_b8 / fwd_b1: one forward pass over prompt + `{"flags":` (as run_knob), batch 8 / 1

`--load hf` loads the model the way run_hf does (`torch_dtype=`), `--load knob` the way run_knob does (`dtype=`);
the dtype actually loaded is recorded. Output: results/diag_knob_<load>.jsonl, compared on the Mac.

Usage (GPU):  python -m fineprint.eval.diag_knob --adapter /kaggle/working/ft/adapter --load hf
"""

import argparse
import json

from fineprint.config import RESULTS
from fineprint.eval.run_knob import EMPTY, FLAG, PREFIX
from fineprint.eval.sets import load_eval_set
from fineprint.prompt import build_messages


def main() -> None:
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--load", choices=("hf", "knob"), required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    prefix = tok(PREFIX, add_special_tokens=False)["input_ids"]
    e_id, f_id = (tok(t, add_special_tokens=False)["input_ids"][0] for t in (EMPTY, FLAG))
    kw = {"torch_dtype": torch.float16} if args.load == "hf" else {"dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", **kw)
    loaded = str(model.dtype)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    meta = {"load": args.load, "loaded_dtype": loaded, "merged_dtype": str(model.dtype),
            "transformers": transformers.__version__, "attn": model.config._attn_implementation}
    print(meta, flush=True)

    todo = sorted(load_eval_set("test"), key=lambda e: len(e["input"]))[: args.n]
    texts = [tok.apply_chat_template(build_messages(e["input"]), tokenize=False, add_generation_prompt=True)
             for e in todo]

    def two_way(logits):
        p = torch.softmax(logits.float(), dim=-1)
        return (p[:, e_id] / (p[:, e_id] + p[:, f_id])).tolist()

    def gen(batch_texts):
        enc = tok(batch_texts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=len(prefix) + 1, do_sample=False, pad_token_id=tok.pad_token_id,
                             output_scores=True, return_dict_in_generate=True)
        new = out.sequences[:, enc["input_ids"].shape[1]:].tolist()
        step = len(prefix)  # scores[step] = distribution for the token after `{"flags":`, if the prefix was followed
        return two_way(out.scores[step]), [tok.convert_ids_to_tokens(s) for s in new]

    def fwd(batch_texts):
        ids = [tok(t, add_special_tokens=False)["input_ids"] + prefix for t in batch_texts]
        enc = tok.pad({"input_ids": ids}, return_tensors="pt").to("cuda")
        return two_way(model(**enc).logits[:, -1, :])

    rows = {e["id"]: {"id": e["id"]} for e in todo}
    with torch.no_grad():
        for bs, tag in ((8, "b8"), (1, "b1")):
            for i in range(0, len(todo), bs):
                part, tx = todo[i:i + bs], texts[i:i + bs]
                g, toks = gen(tx)
                f = fwd(tx)
                for e, gp, fp, tk in zip(part, g, f, toks):
                    rows[e["id"]].update({f"gen_{tag}": round(gp, 6), f"fwd_{tag}": round(fp, 6), f"tokens_{tag}": tk})
            print(tag, "done", flush=True)
    out = RESULTS / f"diag_knob_{args.load}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)  # the Kaggle bundle has no results/ folder
    with open(out, "w") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values())
    print("wrote", out)


if __name__ == "__main__":
    main()
