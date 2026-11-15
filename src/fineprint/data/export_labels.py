"""Export the public labels-only dataset (decision D44): the human-reviewed ToS;DR points the model learned from.

One row per point: quote, our category, severity, explanation (ToS;DR's case title), links back to ToS;DR and the
source document, and the train/validation/test split (same company hash as training, D14). No document text
beyond the quotes, which ToS;DR already publishes; the companies own their documents (DATA.md, licensing).

Usage:  uv run python -m fineprint.data.export_labels   → dist/hf-dataset/{train,validation,test}.parquet
"""

import json

import pandas as pd

from fineprint.config import ROOT, data_path, load_config
from fineprint.data.build_dataset import split_of

OUT = ROOT / "dist" / "hf-dataset"


def main() -> None:
    sp = load_config("data")["split"]
    pts = pd.read_parquet(data_path("interim") / "points_labeled.parquet")
    docs = {}
    for f in (data_path("raw") / "tosdr" / "documents").glob("doc_*.json"):
        d = json.loads(f.read_text())
        d = d.get("parameters", d)
        docs[d["id"]] = d
    names = {}
    for f in (data_path("raw") / "tosdr" / "services").glob("page_*.json"):
        for s in json.loads(f.read_text()).get("services") or []:
            names[s["id"]] = s["name"]
    rows = []
    for r in pts.itertuples():
        doc = docs.get(int(r.document_id)) if pd.notna(r.document_id) else None
        sid = doc.get("service_id") if doc else None
        group = f"service:{sid}" if sid else (f"domain:{r.domain}" if r.domain else f"doc:{r.document_id}")
        rows.append({"point_id": r.point_id, "tosdr_url": f"https://edit.tosdr.org/points/{r.point_id}",
                     "service_id": sid, "service": names.get(sid), "document": (doc or {}).get("name"),
                     "document_url": (doc or {}).get("url"), "quote": r.quote, "category": r.category,
                     "severity": r.severity, "explanation": r.explanation, "case_id": r.case_id,
                     "split": split_of(group, sp["val_fraction"], sp["test_fraction"])})
    df = pd.DataFrame(rows)
    df["service_id"] = df["service_id"].astype("Int64")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        part = df[df.split == split].drop(columns="split")
        part.to_parquet(OUT / f"{name}.parquet", index=False)
        print(name, len(part))


if __name__ == "__main__":
    main()
