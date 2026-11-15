"""Precomputed demo examples (decision D42): ~10 well-known services run through the shipped pipeline once, so
the free CPU Space shows them instantly.

    uv run python -m fineprint.inference.examples prepare
        Mac: raw ToS;DR documents → data/examples/inputs.jsonl (full text; stays out of git, goes in the Kaggle bundle)
    python -m fineprint.inference.examples run --model X.gguf --shard 0/2
        Kaggle CPU: inputs → data/examples/out/<slug>.json (one shard of the examples per kernel)
    uv run python -m fineprint.inference.examples collect
        Mac: out/*.json → app/examples.json, the file the demo reads

The demo file keeps flags, quotes, grade and links, but not the documents' full text: the companies own it, so we
quote short excerpts and link to the source (DATA.md, licensing).
"""

import argparse
import json
import os

from fineprint.config import ROOT, data_path, load_config
from fineprint.data.build_dataset import split_of

EX = ROOT / "data" / "examples"
APP_FILE = ROOT / "app" / "examples.json"


def _services() -> dict[int, dict]:
    out = {}
    for f in (data_path("raw") / "tosdr" / "services").glob("page_*.json"):
        for s in json.loads(f.read_text()).get("services") or []:
            out[s["id"]] = s
    return out


def prepare() -> None:
    svc = _services()
    split_cfg = load_config("data")["split"]
    wanted = {(e["service_id"], e["document"]): e for e in load_config("examples")["examples"]}
    found = {}
    for f in (data_path("raw") / "tosdr" / "documents").glob("doc_*.json"):
        d = json.loads(f.read_text())
        d = d.get("parameters", d)
        e = wanted.get((d.get("service_id"), (d.get("name") or "").strip()))
        if e:
            s = svc[e["service_id"]]
            found[e["slug"]] = {**e, "service": s["name"], "tosdr_rating": s.get("rating"),
                                "tosdr_url": f"https://tosdr.org/en/service/{s['id']}", "url": d.get("url"),
                                "split": split_of(f"service:{s['id']}", split_cfg["val_fraction"],
                                                  split_cfg["test_fraction"]),
                                "text": d.get("text") or ""}
    missing = [e["slug"] for e in wanted.values() if e["slug"] not in found]
    assert not missing, f"documents not found: {missing}"
    EX.mkdir(parents=True, exist_ok=True)
    with open(EX / "inputs.jsonl", "w") as fh:
        for e in load_config("examples")["examples"]:
            fh.write(json.dumps(found[e["slug"]], ensure_ascii=False) + "\n")
    print({k: len(v["text"]) for k, v in found.items()})


def run(model_path: str, shard: str, threads: int | None) -> None:
    from fineprint.inference.gguf import GGUFModel
    from fineprint.inference.pipeline import analyse_document

    i, n = map(int, shard.split("/"))
    rows = [json.loads(line) for line in (EX / "inputs.jsonl").read_text().splitlines()]
    model = GGUFModel(model_path, n_threads=threads)
    (EX / "out").mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(rows):
        if k % n != i:
            continue
        res = analyse_document(r.pop("text"), model.analyse)
        (EX / "out" / f"{r['slug']}.json").write_text(json.dumps({**r, **res}, ensure_ascii=False, indent=1))
        print(r["slug"], res["grade"], len(res["flags"]), "flags", res["n_chunks"], "chunks", res["seconds"], "s",
              flush=True)
    model.close()


def collect() -> None:
    out = []
    for e in load_config("examples")["examples"]:
        r = json.loads((EX / "out" / f"{e['slug']}.json").read_text())
        flags = [{k: f[k] for k in ("category", "severity", "explanation", "quote", "quote_found")}
                 for f in r["flags"]]
        out.append({k: r[k] for k in ("slug", "service", "document", "url", "tosdr_rating", "tosdr_url", "split",
                                      "grade", "score", "counts", "n_chunks", "chars", "seconds")}
                   | {"flags": flags})
    APP_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for r in out:
        print(f"{r['service']:<12} {r['split']:<5} model {r['grade']}  ToS;DR {r['tosdr_rating']}  "
              f"{len(r['flags'])} flags, {r['n_chunks']} chunks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "run", "collect"])
    ap.add_argument("--model")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    {"prepare": prepare, "collect": collect}.get(a.step, lambda: run(a.model, a.shard, a.threads))()


if __name__ == "__main__":
    main()
