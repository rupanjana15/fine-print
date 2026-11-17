"""Build train/val/test JSONL from labelled points + full documents.

Example types (decisions D13, D25, D26):
- positive       a ~200-token window of the real document around an approved quote; flags = every
                 approved quote fully inside the window
- hard_negative  a window around a clause DocBot suggested but a human declined, with no approved
                 quote inside → {"flags": []}
- negative       a regular chunk from a *comprehensively reviewed* service's document that overlaps
                 no approved and no pending (unreviewed) quote → {"flags": []}

Both negative kinds are dropped if any sentence is too similar (TF-IDF) to a known approved quote
from *any* service: ToS;DR records a case about once per service, so a repeat of a known risky
clause is usually just unlabelled, not harmless (decision D31).
- bare           approved quote not found in the current document text (document changed since it
                 was annotated) → the quote alone is the input

Split: by service (D14), using a stable hash so a service never changes split between rebuilds.

Usage:  uv run python -m fineprint.data.build_dataset
Needs:  clean + taxonomy outputs, `fetch_tosdr documents` and `fetch_tosdr services`
Output: data/processed/{train,val,test}.jsonl, results/dataset_report.json
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from contextlib import ExitStack

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from fineprint.chunk import chunk_text, split_sentences, window_around
from fineprint.config import RESULTS, data_path, load_config
from fineprint.data.clean import MAX_CHARS, MIN_CHARS, clean_text
from fineprint.prompt import build_messages


def load_documents() -> dict[int, dict]:
    docs = {}
    for f in (data_path("raw") / "tosdr" / "documents").glob("doc_*.json"):
        data = json.loads(f.read_text())
        if data.get("_missing"):
            continue
        d = data.get("parameters", data)
        if d.get("text"):
            docs[d["id"]] = {"service_id": d.get("service_id"), "text": clean_text(d["text"])}
    return docs


def load_comprehensive_services() -> set[int]:
    pages = list((data_path("raw") / "tosdr" / "services").glob("page_*.json"))
    if not pages:
        raise SystemExit("Service list missing: run `uv run python -m fineprint.data.fetch_tosdr services`")
    ids = set()
    for f in pages:
        for s in json.loads(f.read_text()).get("services") or []:
            if s.get("is_comprehensively_reviewed"):
                ids.add(s["id"])
    return ids


def max_quote_similarity(texts: list[str], quotes: list[str], batch: int = 5000) -> list[float]:
    """For each text, the highest TF-IDF cosine between any of its sentences and any quote."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")
    q = vec.fit_transform(quotes).T.tocsc()
    owner, sents = [], []
    for i, t in enumerate(texts):
        for s, e in split_sentences(t, 800):
            owner.append(i)
            sents.append(t[s:e])
    best = [0.0] * len(texts)
    for b in range(0, len(sents), batch):
        scores = (vec.transform(sents[b:b + batch]) @ q).max(axis=1).toarray().ravel()
        for i, v in zip(owner[b:b + batch], scores):
            best[i] = max(best[i], float(v))
    return best


def split_of(group: str, val_fraction: float, test_fraction: float) -> str:
    """Stable split from a hash of the group, independent of data order or size."""
    bucket = int(hashlib.md5(group.encode()).hexdigest(), 16) % 1000 / 1000
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def find_spans(text: str, quotes: list[str]) -> list[tuple[int, int] | None]:
    out = []
    for q in quotes:
        i = text.find(q)
        out.append((i, i + len(q)) if i >= 0 else None)
    return out


def overlaps(a: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(s < a[1] and a[0] < e for s, e in spans)


def flags_inside(window: tuple[int, int], labeled: list[dict]) -> list[dict]:
    """Approved flags whose quote lies fully inside the window, in reading order, deduplicated."""
    seen, flags = set(), []
    for item in sorted(labeled, key=lambda x: x["span"]):
        s, e = item["span"]
        key = (item["category"], item["explanation"], item["quote"])
        if window[0] <= s and e <= window[1] and key not in seen:
            seen.add(key)
            flags.append({k: item[k] for k in ("category", "severity", "explanation", "quote")})
    return flags


def candidate_quotes(points: pd.DataFrame) -> pd.DataFrame:
    """Clean quotes of non-approved points, used to exclude/choose negative regions."""
    q = points.assign(quote=points["quote_text"].fillna("").map(clean_text))
    return q[q["quote"].str.len().between(MIN_CHARS, MAX_CHARS)]


def main() -> None:
    cfg = load_config("data")
    chunk_cfg, split_cfg = cfg["chunking"], cfg["split"]
    rng = random.Random(split_cfg["seed"])

    labeled = pd.read_parquet(data_path("interim") / "points_labeled.parquet")
    all_points = pd.read_parquet(data_path("interim") / "points.parquet")
    pending = candidate_quotes(all_points[all_points["status"] == "pending"])
    declined_docbot = candidate_quotes(
        all_points[(all_points["status"] == "declined") & all_points["ml_score"].notna()])
    docs = load_documents()
    comprehensive = load_comprehensive_services()
    report = Counter(documents_loaded=len(docs), comprehensive_services=len(comprehensive))

    # A declined DocBot suggestion is only a trustworthy negative if the document has NO approved
    # point for that case. Otherwise it was most likely declined as a duplicate of an approved
    # point, i.e. the clause *is* about that case (seen in the v0.2 sample review).
    approved_doc_case = set(zip(labeled["document_id"], labeled["case_id"]))
    is_dup = [(d, c) in approved_doc_case
              for d, c in zip(declined_docbot["document_id"], declined_docbot["case_id"])]
    report["declined_docbot_total"] = len(declined_docbot)
    report["declined_docbot_skipped_case_already_approved"] = sum(is_dup)
    declined_docbot = declined_docbot[[not x for x in is_dup]]

    # Index once by document (filtering big tables per document was the slow part).
    labeled_quotes_by_doc = labeled.groupby("document_id")["quote"].apply(list).to_dict()
    pending_by_doc = pending.groupby("document_id")["quote"].apply(list).to_dict()
    declined_by_doc = declined_docbot.groupby("document_id")["quote"].apply(list).to_dict()

    examples: list[dict] = []
    seen_inputs: set[str] = set()

    def add(kind: str, text: str, flags: list[dict], group: str, doc_id) -> None:
        if text in seen_inputs:
            report[f"duplicate_{kind}"] += 1
            return
        seen_inputs.add(text)
        examples.append({"kind": kind, "group": group, "doc_id": doc_id, "text": text, "flags": flags})

    by_doc = defaultdict(list)
    for r in labeled.itertuples():
        by_doc[r.document_id].append(r)

    for doc_id, rows in by_doc.items():
        doc = docs.get(int(doc_id)) if pd.notna(doc_id) else None
        group = (f"service:{doc['service_id']}" if doc and doc["service_id"]
                 else f"domain:{rows[0].domain}" if rows[0].domain else f"doc:{doc_id}")

        if doc is None:
            for r in rows:
                add("bare", r.quote, [{"category": r.category, "severity": r.severity,
                                            "explanation": r.explanation, "quote": r.quote}], group, doc_id)
            report["points_without_document"] += len(rows)
            continue

        text = doc["text"]
        spans = find_spans(text, [r.quote for r in rows])
        located = []
        for r, span in zip(rows, spans):
            flag = {"category": r.category, "severity": r.severity, "explanation": r.explanation, "quote": r.quote}
            if span is None:
                add("bare", r.quote, [flag], group, doc_id)
                report["points_quote_not_in_document"] += 1
            else:
                located.append({**flag, "span": span})
                report["points_located"] += 1
        approved_spans = [x["span"] for x in located]

        for item in located:
            s, e = window_around(text, *item["span"], chunk_cfg["target_tokens"], rng)
            add("positive", text[s:e], flags_inside((s, e), located), group, doc_id)

        for span in find_spans(text, declined_by_doc.get(doc_id, [])):
            if span is None or overlaps(span, approved_spans):
                continue
            s, e = window_around(text, *span, chunk_cfg["target_tokens"], rng)
            flags = flags_inside((s, e), located)
            add("positive" if flags else "hard_negative", text[s:e], flags, group, doc_id)

    # Easy negatives: only from comprehensively reviewed services, where unflagged text is
    # much more likely to really be harmless (not just un-annotated).
    labeled_docs = set(labeled["document_id"].dropna().astype(int))
    for doc_id in labeled_docs:
        doc = docs.get(doc_id)
        if not doc or doc["service_id"] not in comprehensive:
            continue
        text = doc["text"]
        blocked = [x for x in find_spans(text, labeled_quotes_by_doc.get(doc_id, [])) if x]
        blocked += [x for x in find_spans(text, pending_by_doc.get(doc_id, [])) if x]
        for s, e in chunk_text(text, chunk_cfg["target_tokens"], chunk_cfg["overlap_tokens"]):
            if e - s >= 200 and not overlaps((s, e), blocked):
                add("negative", text[s:e], [], f"service:{doc['service_id']}", doc_id)

    neg = [ex for ex in examples if not ex["flags"] and ex["kind"] != "bare"]
    limit = cfg["negatives"]["max_quote_similarity"]
    for ex, sim in zip(neg, max_quote_similarity([ex["text"] for ex in neg], labeled["quote"].tolist())):
        if sim >= limit:
            ex["drop"] = True
            report[f"{ex['kind']}_dropped_similar_to_known_quote"] += 1
    examples = [ex for ex in examples if not ex.get("drop")]

    for ex in examples:
        ex["split"] = split_of(ex["group"], split_cfg["val_fraction"], split_cfg["test_fraction"])

    # Balance: in each split, negatives (hard first, then easy) fill up to target_fraction.
    frac = cfg["negatives"]["target_fraction"]
    final = []
    for split in ("train", "val", "test"):
        part = [e for e in examples if e["split"] == split]
        pos = [e for e in part if e["flags"]]
        hard = [e for e in part if e["kind"] == "hard_negative"]
        easy = [e for e in part if e["kind"] == "negative"]
        need = round(len(pos) * frac / (1 - frac))
        rng.shuffle(hard)
        rng.shuffle(easy)
        hard = hard[: need // 2]  # at most half of negatives are hard: keep plenty of ordinary text
        chosen = pos + hard + easy[: need - len(hard)]
        rng.shuffle(chosen)
        final += chosen
        report[f"{split}_examples"] = len(chosen)
        report[f"{split}_positive"] = len(pos)
        report[f"{split}_hard_negative"] = len(hard)
        report[f"{split}_negative"] = len(chosen) - len(pos) - len(hard)
        report[f"{split}_groups"] = len({e["group"] for e in chosen})

    out_dir = data_path("processed")
    with ExitStack() as stack:
        files = {s: stack.enter_context(open(out_dir / f"{s}.jsonl", "w")) for s in ("train", "val", "test")}
        for i, ex in enumerate(final):
            record = {"id": f"{ex['split']}-{i:06d}", "kind": ex["kind"], "group": ex["group"],
                      "doc_id": None if pd.isna(ex["doc_id"]) else int(ex["doc_id"]),
                      "messages": build_messages(ex["text"], ex["flags"])}
            files[ex["split"]].write(json.dumps(record, ensure_ascii=False) + "\n")

    pos_flags = Counter(f["category"] for e in final if e["split"] == "train" for f in e["flags"])
    report = dict(report)
    report["train_flags_per_category"] = dict(pos_flags.most_common())
    report["flags_per_positive_example"] = round(
        sum(len(e["flags"]) for e in final if e["flags"]) / max(1, sum(1 for e in final if e["flags"])), 2)
    (RESULTS / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "train_flags_per_category"}, indent=2))


if __name__ == "__main__":
    main()
