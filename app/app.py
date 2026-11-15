"""Fine Print demo (v0.7, decisions D19, D20, D42): Gradio on a free CPU Hugging Face Space.

Two tabs: precomputed results for well-known services (instant), and "paste your own terms", which runs the shipped
4-bit GGUF through the same pipeline as everything else (`fineprint.inference.pipeline`), with a length cap because
a free 2-vCPU Space needs several seconds per chunk.

Run locally:  uv run --extra app python app/app.py        (model from FINEPRINT_MODEL, else the HF Hub)
"""

import atexit
import html
import json
import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # repo layout locally and on the Space: <root>/src/fineprint

from fineprint.chunk import chunk_text
from fineprint.config import load_config
from fineprint.data.clean import clean_text
from fineprint.inference.pipeline import analyse_document

MAX_CHARS = 6000  # ~9 chunks: a few minutes on the free CPU Space
MODEL_REPO = os.environ.get("FINEPRINT_MODEL_REPO", "rupp15/fine-print-gguf")
MODEL_FILE = "fine-print-v2-q4_k_m.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}"
CATEGORIES = load_config("taxonomy")["categories"]
GRADE_PARAMS = load_config("grade")
EXAMPLES = {e["service"]: e for e in json.loads((HERE / "examples.json").read_text())}
SEV = {"blocker": ("Blocker", "#b42318"), "bad": ("Bad", "#d9480f"), "neutral": ("Neutral", "#667085"),
       "good": ("Good", "#2f9e44")}
GRADE_COLOUR = {"A": "#2f9e44", "B": "#74b816", "C": "#f59f00", "D": "#e8590c", "E": "#c92a2a"}
DISCLAIMER = ("**Fine Print estimate, not legal advice.** The grade is computed by a small model and a counting "
              "rule; it is not ToS;DR's grade. For human-reviewed ratings see [tosdr.org](https://tosdr.org).")
RULE = (f"Grade rule: good +{GRADE_PARAMS['w_good']:g}, bad −1, blocker −{GRADE_PARAMS['w_blocker']:g} per distinct "
        f"issue (neutral flags don't count); A ≥ {GRADE_PARAMS['cuts'][0]:g}, B ≥ {GRADE_PARAMS['cuts'][1]:g}, "
        f"C ≥ {GRADE_PARAMS['cuts'][2]:g}, D ≥ {GRADE_PARAMS['cuts'][3]:g}, else E. Fitted on ToS;DR's human grades: "
        "exact on 58% of unseen companies, never more than one grade off.")

_model = None


def model():
    global _model
    if _model is None:
        from fineprint.inference.gguf import GGUFModel

        path = os.environ.get("FINEPRINT_MODEL")
        if not path:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        _model = GGUFModel(path, n_threads=os.cpu_count())
        atexit.register(_model.close)  # freeing the model before exit avoids a Metal teardown abort on Macs
    return _model


def grade_card(r: dict, extra: str = "") -> str:
    c = r["counts"]
    g = r["grade"]
    return (f'<div style="display:flex;gap:16px;align-items:center">'
            f'<div style="font-size:48px;font-weight:700;color:white;background:{GRADE_COLOUR[g]};border-radius:12px;'
            f'width:72px;height:72px;display:flex;align-items:center;justify-content:center">{g}</div>'
            f'<div><b>Fine Print estimate: {g}</b> (score {r["score"]:g})<br>'
            f'{c["blocker"]} blocker · {c["bad"]} bad · {c["good"]} good · {c["neutral"]} neutral issues'
            f'{extra}</div></div>')


def flag_list(flags: list[dict]) -> str:
    order = {"blocker": 0, "bad": 1, "good": 2, "neutral": 3, None: 4}
    rows = []
    for f in sorted(flags, key=lambda f: order.get(f["severity"], 4)):
        name, colour = SEV.get(f["severity"], ("?", "#999"))
        rows.append(
            f'<div style="border-left:4px solid {colour};padding:6px 12px;margin:8px 0">'
            f'<span style="color:{colour};font-weight:600">{name}</span> · '
            f'<span title="{html.escape(CATEGORIES.get(f["category"], ""))}">{f["category"].replace("_", " ")}</span>'
            f'<br>{html.escape(f["explanation"])}'
            f'<br><i style="opacity:.75">“{html.escape(f["quote"])}”</i></div>')
    return "".join(rows) or "<p>No clauses flagged.</p>"


def show_example(name: str):
    e = EXAMPLES[name]
    seen = ("The model never saw this company's terms in training."
            if e["split"] in ("val", "test") else "This company's terms were part of the training data.")
    extra = (f'<br>ToS;DR\'s human-reviewed grade for the whole service: <b>{e["tosdr_rating"]}</b> '
             f'(<a href="{e["tosdr_url"]}" target="_blank">tosdr.org</a>)'
             f'<br>Document: <a href="{e["url"]}" target="_blank">{html.escape(e["document"])}</a>, '
             f'{e["n_chunks"]} chunks. {seen}')
    return grade_card(e, extra), flag_list(e["flags"])


def analyse_text(text: str, progress=gr.Progress()):  # noqa: B008 (Gradio injects progress this way)
    text = (text or "").strip()
    if not text:
        raise gr.Error("Paste some terms of service first.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"That's {len(text):,} characters; the free demo takes up to {MAX_CHARS:,}. "
                       "Paste one section, or run it locally (see the README).")
    cfg = load_config("data")["chunking"]
    n = len(chunk_text(clean_text(text), cfg["target_tokens"], cfg["overlap_tokens"]))
    progress(0, desc="Loading the model")
    m = model()
    done = [0]

    def predict(chunk: str) -> dict:
        progress(done[0] / n, desc=f"Reading chunk {done[0] + 1} of {n}")
        done[0] += 1
        return m.analyse(chunk)

    r = analyse_document(text, predict)
    sev_of = {}
    for f in r["flags"]:
        if f["span"] and f["severity"]:
            sev_of[tuple(f["span"])] = SEV[f["severity"]][0]
    clean = clean_text(text)
    segs, pos = [], 0
    for (s, e), label in sorted(sev_of.items()):
        if s < pos:
            continue
        segs += [(clean[pos:s], None), (clean[s:e], label)]
        pos = e
    segs.append((clean[pos:], None))
    segs = [(t, lab) for t, lab in segs if t]
    return grade_card(r, f"<br>{r['n_chunks']} chunks in {r['seconds']:.0f} s"), segs, flag_list(r["flags"])


with gr.Blocks(title="Fine Print") as demo:
    gr.Markdown("# Fine Print\nA fine-tuned 3B model that reads Terms of Service and flags the clauses that matter, "
                "with severity, a plain-English explanation and the exact quote.\n\n" + DISCLAIMER)
    with gr.Tab("Well-known services"):
        gr.Markdown("**How far to trust the grade:** on these 10 services the estimate matched ToS;DR's human grade "
                    "for 1 and was within one grade for 5; for 8 of 10 it was more generous. It reads one document, "
                    "while ToS;DR grades the whole service (privacy policy included), and the model tends to call "
                    "some clauses neutral that ToS;DR's reviewers call bad. The individual flags are the more "
                    "reliable part.")
        pick = gr.Dropdown(list(EXAMPLES), value=next(iter(EXAMPLES)), label="Service")
        card, flags = gr.HTML(), gr.HTML()
        pick.change(show_example, pick, [card, flags])
        demo.load(show_example, pick, [card, flags])
    with gr.Tab("Paste your own"):
        box = gr.Textbox(lines=12, label=f"Terms of Service text (up to {MAX_CHARS:,} characters)")
        go = gr.Button("Analyse", variant="primary")
        card2 = gr.HTML()
        hl = gr.HighlightedText(label="Flagged clauses", show_legend=True,
                                color_map={v[0]: v[1] for v in SEV.values()})
        flags2 = gr.HTML()
        go.click(analyse_text, box, [card2, hl, flags2])
    gr.Markdown(RULE + " Precomputed examples cover one main document per service; ToS;DR grades the whole service. "
                "Data: [ToS;DR](https://tosdr.org) (CC BY-SA 3.0). **Built with Qwen**: Qwen2.5-3B-Instruct fine-tuned with "
                f"QLoRA, non-commercial use under the Qwen Research License ([model]({MODEL_URL})).")

if __name__ == "__main__":
    demo.launch()
