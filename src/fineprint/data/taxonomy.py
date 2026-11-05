"""Map cleaned ToS;DR points to our categories (configs/taxonomy.yaml, decision D24).

Usage:  uv run python -m fineprint.data.taxonomy      (after clean)
Input:  data/interim/points_clean.parquet
Output: data/interim/points_labeled.parquet   adds: category, explanation
        results/taxonomy_report.json          per-category counts and severity mix
"""

import json

import pandas as pd

from fineprint.config import RESULTS, data_path, load_config


def load_taxonomy() -> dict:
    tax = load_config("taxonomy")
    unknown = {c for c in [*tax["topics"].values(), *tax["case_overrides"].values()]
               if c not in tax["categories"]}
    if unknown:
        raise ValueError(f"Mapped to undefined categories: {unknown}")
    return tax


def assign_category(case_id: int, topic_id: int, tax: dict) -> str | None:
    if case_id in tax["case_overrides"]:
        return tax["case_overrides"][case_id]
    return tax["topics"].get(topic_id)


def explanation_from_title(title: str) -> str:
    """Case titles are the plain-English explanation; tidy stray whitespace/markers."""
    return " ".join(title.replace("[DEPRECATED]", "").split())


def label_points(df: pd.DataFrame, tax: dict) -> tuple[pd.DataFrame, dict]:
    report = {"input_points": len(df)}

    df = df[~df["case_id"].isin(tax["drop_cases"])]
    report["removed_dropped_cases"] = report["input_points"] - len(df)

    counts = df["case_id"].value_counts()
    rare = counts[counts < tax["min_examples_per_case"]].index
    report["removed_rare_cases"] = {"cases": len(rare), "points": int(df["case_id"].isin(rare).sum())}
    df = df[~df["case_id"].isin(rare)]

    df = df.assign(
        category=[assign_category(c, t, tax) for c, t in zip(df["case_id"], df["topic_id"])],
        explanation=df["case_title"].map(explanation_from_title),
    )
    unmapped = df["category"].isna()
    report["removed_unmapped_topic"] = int(unmapped.sum())
    df = df[~unmapped].reset_index(drop=True)

    report["output_points"] = len(df)
    report["output_cases"] = int(df["case_id"].nunique())
    report["categories"] = {
        cat: {"points": len(g), "cases": int(g["case_id"].nunique()),
              "severity": g["severity"].value_counts().to_dict()}
        for cat, g in sorted(df.groupby("category"), key=lambda kv: -len(kv[1]))
    }
    return df, report


def main() -> None:
    tax = load_taxonomy()
    df = pd.read_parquet(data_path("interim") / "points_clean.parquet")
    labeled, report = label_points(df, tax)
    labeled.to_parquet(data_path("interim") / "points_labeled.parquet")
    (RESULTS / "taxonomy_report.json").write_text(json.dumps(report, indent=2))

    print(f"{report['output_points']:,} points, {report['output_cases']} cases, "
          f"{len(report['categories'])} categories "
          f"(dropped: {report['removed_rare_cases']}, unmapped: {report['removed_unmapped_topic']})\n")
    for cat, r in report["categories"].items():
        sev = " ".join(f"{k[:3]}={v}" for k, v in sorted(r["severity"].items()))
        print(f"  {cat:<36} {r['points']:>5} pts {r['cases']:>3} cases  {sev}")


if __name__ == "__main__":
    main()
