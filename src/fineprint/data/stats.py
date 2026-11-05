"""Turn the raw ToS;DR cache into one table and print the numbers we need for v0.2 decisions.

Usage:  uv run python -m fineprint.data.stats
Output: data/interim/points.parquet       one row per point (third-party text, not committed)
        data/interim/sample_100.md        100 random usable points to read by hand
        results/data_stats.json           aggregate counts only (committed)
"""

import json
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

from fineprint.config import RESULTS, data_path

# Rough English check: share of tokens that are very common English function words.
_EN_WORDS = set(
    "the of and to a in that is for it as with be on by this are or not you your we our may "
    "any will from at have an if can such use which all their they us information".split()
)


def looks_english(text: str, threshold: float = 0.15) -> bool:
    tokens = re.findall(r"[a-z]+", text.lower())
    if len(tokens) < 5:
        return False
    return sum(t in _EN_WORDS for t in tokens) / len(tokens) >= threshold


def source_domain(url: str | None) -> str | None:
    """'https://www.facebook.com/policy' -> 'facebook.com' (used as a stand-in for the service)."""
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        host = urlparse(url if "://" in url else f"https://{url.strip()}").hostname or ""
    except ValueError:  # malformed URLs, e.g. broken IPv6 brackets
        return None
    return host.removeprefix("www.") or None


def load_points() -> pd.DataFrame:
    rows = []
    for f in sorted((data_path("raw") / "tosdr" / "points_by_case").glob("case_*_page_*.json")):
        data = json.loads(f.read_text())
        for p in data.get("points") or []:
            case = p.get("case") or {}
            rows.append({
                "point_id": p["id"],
                "status": p.get("status"),
                "title": p.get("title"),
                "analysis": p.get("analysis"),
                "quote_text": p.get("quote_text"),
                "quote_start": p.get("quote_start"),
                "quote_end": p.get("quote_end"),
                "document_id": p.get("document_id"),
                "source": p.get("source"),
                "case_id": case.get("id"),
                "case_title": case.get("title"),
                "case_classification": case.get("classification"),
                "case_weight": case.get("weight"),
                "topic_id": case.get("topic_id"),
            })
    df = pd.DataFrame(rows).drop_duplicates("point_id")
    df["domain"] = df["source"].map(source_domain)
    df["has_quote"] = df["quote_text"].fillna("").str.strip().str.len() > 0
    df["quote_chars"] = df["quote_text"].fillna("").str.len()
    df["english"] = df["quote_text"].fillna("").map(looks_english)
    df["usable"] = (df["status"] == "approved") & df["has_quote"] & df["english"]
    return df


def main() -> None:
    df = load_points()
    interim = data_path("interim")
    df.to_parquet(interim / "points.parquet")
    use = df[df["usable"]]

    per_case = use.groupby(["case_id", "case_title", "case_classification"]).size()
    per_case = per_case.sort_values(ascending=False)
    per_domain = use["domain"].value_counts()

    stats = {
        "points_total": len(df),
        "cases_total": int(df["case_id"].nunique()),
        "status": df["status"].value_counts().to_dict(),
        "approved_with_quote": int(((df["status"] == "approved") & df["has_quote"]).sum()),
        "usable_english_approved_with_quote": len(use),
        "usable_classification": use["case_classification"].value_counts().to_dict(),
        "usable_quote_chars": use["quote_chars"].describe(percentiles=[.1, .5, .9]).round(0).to_dict(),
        "usable_cases": int(per_case.size),
        "cases_with_at_least": {n: int((per_case >= n).sum()) for n in (20, 50, 100, 200, 500)},
        "usable_domains": int(per_domain.size),
        "usable_documents": int(use["document_id"].nunique()),
        "top_domains": per_domain.head(15).to_dict(),
        "top_cases": [
            {"case_id": int(cid), "title": title, "classification": cls, "n": int(n)}
            for (cid, title, cls), n in per_case.head(60).items()
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "data_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    print(f"Points: {stats['points_total']:,} across {stats['cases_total']} cases")
    print(f"Status: {stats['status']}")
    print(f"Approved + quote: {stats['approved_with_quote']:,} | + English: {len(use):,}")
    print(f"Severity: {stats['usable_classification']}")
    print(f"Quote length (chars): {stats['usable_quote_chars']}")
    print(f"Usable cases: {stats['usable_cases']} | with >=N examples: {stats['cases_with_at_least']}")
    print(f"Domains: {stats['usable_domains']:,} | documents: {stats['usable_documents']:,}")
    print("\nTop 25 cases:")
    for c in stats["top_cases"][:25]:
        print(f"  {c['n']:>5}  {c['classification']:<8} [{c['case_id']:>4}] {c['title'][:80]}")

    sample = use.sample(min(100, len(use)), random_state=42)
    lines = ["# 100 random usable points — read these by hand\n"]
    for _, r in sample.iterrows():
        lines.append(f"## [{r.case_classification}] {r.case_title}\n"
                     f"*point {r.point_id} · {r.domain}*\n\n> {r.quote_text.strip()}\n")
    (interim / "sample_100.md").write_text("\n".join(lines))
    print(f"\nWrote {interim / 'sample_100.md'} and {RESULTS / 'data_stats.json'}")


if __name__ == "__main__":
    main()
