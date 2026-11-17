"""Clean ToS;DR points into a training-ready table.

Keeps human-approved points (D07) with a quote, strips HTML, keeps English (D15), drops
duplicates and extreme lengths.

Usage:  uv run python -m fineprint.data.clean       (after fetch_tosdr points + stats)
Input:  data/interim/points.parquet
Output: data/interim/points_clean.parquet
        results/clean_report.json   (how many rows each filter removed)
"""

import html
import json
import re

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from fineprint.config import RESULTS, data_path
from fineprint.data.stats import looks_english

DetectorFactory.seed = 0  # langdetect is random by default; fix it for reproducibility

MIN_CHARS = 20
MAX_CHARS = 1500  # longer "quotes" are whole sections, not clauses, and exceed one chunk

_BLOCK_TAGS = re.compile(r"</?(p|div|li|ul|ol|h[1-6]|br|tr|td|table|section)\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]*>")
_TRAILING_PARTIAL_TAG = re.compile(r"<[^>]*$")
_LEADING_PARTIAL_TAG = re.compile(r"^[^<>]{0,60}>")


def clean_text(text: str) -> str:
    """Strip HTML tags/entities and normalise whitespace, keeping paragraph breaks as spaces."""
    text = _BLOCK_TAGS.sub(" ", text)
    text = _ANY_TAG.sub("", text)
    text = _TRAILING_PARTIAL_TAG.sub("", text)  # quote cut mid-tag, e.g. "...account.</li>\n</"
    text = _LEADING_PARTIAL_TAG.sub("", text)   # quote starting mid-tag, e.g. 'class="x">We may'
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    """langdetect for normal text; the stop-word heuristic for very short text, where langdetect is noisy."""
    if len(text) < 40:
        return looks_english(text, threshold=0.1)
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def main() -> None:
    df = pd.read_parquet(data_path("interim") / "points.parquet")
    report = {"input_points": len(df)}

    def step(name: str, mask: pd.Series) -> pd.DataFrame:
        report[name] = int((~mask).sum())
        return df[mask]

    df = step("removed_not_approved", df["status"] == "approved")
    df = step("removed_no_quote", df["quote_text"].fillna("").str.strip().str.len() > 0)

    df = df.assign(quote=df["quote_text"].map(clean_text))
    df = step("removed_too_short", df["quote"].str.len() >= MIN_CHARS)
    df = step("removed_too_long", df["quote"].str.len() <= MAX_CHARS)
    df = step("removed_not_english", df["quote"].map(is_english))

    before = len(df)
    df = df.drop_duplicates(["quote", "case_id"])
    report["removed_duplicate_quote_case"] = before - len(df)

    out = df.rename(columns={"case_classification": "severity", "quote_text": "quote_raw"})[[
        "point_id", "case_id", "case_title", "severity", "topic_id", "document_id",
        "domain", "quote", "quote_raw", "quote_start", "quote_end",
    ]].reset_index(drop=True)

    report["output_points"] = len(out)
    report["output_cases"] = int(out["case_id"].nunique())
    report["quotes_shared_by_several_cases"] = int(out.groupby("quote")["case_id"].nunique().gt(1).sum())
    report["severity"] = out["severity"].value_counts().to_dict()

    out.to_parquet(data_path("interim") / "points_clean.parquet")
    (RESULTS / "clean_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
