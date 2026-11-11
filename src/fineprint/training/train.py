"""QLoRA fine-tune of the base model on dataset v1 (decisions D04, D37).

- Each example = the shared chat prompt (prompt.py) + the canonical JSON answer. The loss is computed on the
  answer tokens only ("completion-only"): prompt tokens get label -100, so the model is never trained to
  reproduce the ToS text, only to answer about it.
- Examples longer than `max_seq_length` are dropped, never truncated (a cut answer would teach broken JSON).
- Base model frozen in 4-bit (NF4); LoRA adapters train in higher precision; fp16 mixed precision on a T4
  (fp32 LoRA weights + fp16 maths, see CONCEPTS "Why training needs fp32").
- Two backends, compared by a timing run (D37): `hf` = transformers + peft + bitsandbytes, works with
  1 GPU or 2 GPUs via torchrun (DDP); `unsloth` = Unsloth's patched kernels, 1 GPU.
- A time budget stops training cleanly and saves, so a Kaggle session never dies mid-run without output.

Usage:
    # Kaggle, 2 GPUs:   torchrun --nproc_per_node 2 -m fineprint.training.train --out /kaggle/working/run1
    # Kaggle, Unsloth:  python -m fineprint.training.train --backend unsloth --out ...
    # Mac smoke test:   uv run --extra train python -m fineprint.training.train --no-4bit \
    #                     --model Qwen/Qwen2.5-0.5B-Instruct --limit 16 --max-steps 2 --out /tmp/ft
"""

import argparse
import inspect
import json
import math
import os
import time

from fineprint.config import load_config
from fineprint.eval.sets import _processed
from fineprint.prompt import build_messages


def tokenize_example(tok, ex: dict, max_len: int) -> dict | None:
    """input_ids for prompt + answer; labels = -100 on the prompt part. None if too long."""
    messages = build_messages(ex["input"], ex["flags"])
    prompt = tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    # Tokenize the text ourselves: in transformers v5 apply_chat_template(tokenize=True) returns a dict.
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:
        raise ValueError(f"prompt is not a token prefix of the full example ({ex['id']})")
    if len(f_ids) > max_len:
        return None
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids),
            "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]}


def build_split(tok, split: str, max_len: int, limit: int | None = None) -> tuple[list[dict], int]:
    rows = _processed(split)[:limit]
    out = [tokenize_example(tok, ex, max_len) for ex in rows]
    kept = [r for r in out if r is not None]
    return kept, len(out) - len(kept)


def load_model(args, cfg, local_rank: int):
    import torch

    lora = cfg["lora"]
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(
            args.model, max_seq_length=cfg["max_seq_length"], load_in_4bit=not args.no_4bit, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], use_gradient_checkpointing="unsloth",
            random_state=cfg["training"]["seed"])
        return model, tok

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.no_4bit:  # CPU/Mac smoke test only
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, dtype=torch.float16,
                                                     device_map={"": local_rank})
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
    return model, tok


def main() -> None:
    cfg = load_config("train")
    tr = cfg["training"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder (checkpoints, adapter, summary)")
    ap.add_argument("--backend", choices=["hf", "unsloth"], default="hf")
    ap.add_argument("--model", default=cfg["base_model"])
    ap.add_argument("--epochs", type=float, default=tr["num_epochs"])
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (timing run)")
    ap.add_argument("--max-hours", type=float, default=tr["max_hours"], help="stop + save after this long")
    ap.add_argument("--limit", type=int, help="only the first N train examples (smoke test / subsample)")
    ap.add_argument("--val-size", type=int, default=tr["val_size"])
    ap.add_argument("--resume", help="checkpoint folder to resume from")
    ap.add_argument("--no-4bit", action="store_true", help="CPU smoke test without bitsandbytes")
    args = ap.parse_args()

    if args.backend == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers so its patches apply)
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    main_proc = local_rank == 0
    t_start = time.time()

    model, tok = load_model(args, cfg, local_rank)
    tok.padding_side = "right"  # training pads on the right; generation (run_hf) pads on the left
    max_len = cfg["max_seq_length"]
    train, dropped = build_split(tok, "train", max_len, args.limit)
    val, _ = build_split(tok, "val", max_len, None)
    val = val[: args.val_size]  # val loss only, a fixed slice keeps it comparable across runs
    if main_proc:
        model.print_trainable_parameters()
        print(f"train {len(train)} examples ({dropped} dropped > {max_len} tokens), val {len(val)}, "
              f"world size {world}", flush=True)

    # Keep the effective batch (per-device × accumulation × GPUs) at the configured 16 on 1 or 2 GPUs.
    accum = max(1, tr["gradient_accumulation_steps"] // world)
    cuda = torch.cuda.is_available()

    class TimeBudget(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if time.time() - t_start > args.max_hours * 3600:
                control.should_save = True
                control.should_training_stop = True

    # Installing Unsloth can pull in an older transformers (v4), which names two settings differently.
    params = inspect.signature(TrainingArguments.__init__).parameters
    compat = ({"train_sampling_strategy": "group_by_length"} if "train_sampling_strategy" in params
              else {"group_by_length": True})  # similar lengths per batch → less padding
    compat |= ({"warmup_steps": tr["warmup_ratio"]} if "warmup_ratio" not in params
               else {"warmup_ratio": tr["warmup_ratio"]})
    targs = TrainingArguments(
        output_dir=args.out, seed=tr["seed"], report_to="none",
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=tr["per_device_batch_size"],
        per_device_eval_batch_size=2,  # val loss needs logits over the 151k vocab: batch 8 ran out of memory (D37)
        gradient_accumulation_steps=accum, learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler"], weight_decay=0.0, **compat,
        fp16=cuda, bf16=False,  # T4 has no bf16: fp16 mixed precision (LoRA weights stay fp32)
        logging_steps=10, eval_strategy="steps", eval_steps=tr["save_steps"],
        save_strategy="steps", save_steps=tr["save_steps"], save_total_limit=2,
        ddp_find_unused_parameters=False if world > 1 else None,
        gradient_checkpointing=args.backend == "hf" and not args.no_4bit,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=2 if cuda else 0,
        use_cpu=not cuda,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train, eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[TimeBudget()],
    )
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume)
    secs = time.time() - t0
    final_eval = trainer.evaluate()

    if main_proc:
        trainer.save_model(f"{args.out}/adapter")  # LoRA weights only (~100 MB), not the base model
        tok.save_pretrained(f"{args.out}/adapter")
        steps = trainer.state.global_step
        examples = steps * tr["per_device_batch_size"] * accum * world
        mean_len = sum(len(r["input_ids"]) for r in train) / len(train)
        summary = {
            "backend": args.backend, "model": args.model, "gpus": world, "steps": steps,
            "epochs_done": round(examples / len(train), 3), "train_examples": len(train), "dropped_long": dropped,
            "seconds": round(secs), "seconds_per_step": round(secs / max(steps, 1), 2),
            "tokens_per_second": round(examples * mean_len / secs),
            "hours_per_epoch_est": round(len(train) / max(examples, 1) * secs / 3600, 2),
            "train_loss": round(result.training_loss, 4), "val_loss": round(final_eval["eval_loss"], 4),
            "log_history": trainer.state.log_history,
        }
        with open(f"{args.out}/train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2), flush=True)
        if not math.isfinite(summary["val_loss"]):
            raise SystemExit("val loss is not finite")


if __name__ == "__main__":
    main()
