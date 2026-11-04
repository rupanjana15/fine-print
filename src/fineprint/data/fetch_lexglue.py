"""Download LexGLUE unfair_tos (CLAUDETTE) and print label statistics.

Usage:  uv run python -m fineprint.data.fetch_lexglue
Output: data/raw/lexglue/{train,validation,test}.parquet
"""

from collections import Counter

from datasets import load_dataset

from fineprint.config import data_path, load_config


def main() -> None:
    cfg = load_config("data")["lexglue"]
    out_dir = data_path("raw") / "lexglue"
    out_dir.mkdir(exist_ok=True)

    ds = load_dataset(cfg["hf_name"], cfg["config"])
    label_names = ds["train"].features["labels"].feature.names

    for split, part in ds.items():
        df = part.to_pandas()
        df["label_names"] = df["labels"].apply(lambda ids: [label_names[i] for i in ids])
        df.to_parquet(out_dir / f"{split}.parquet")

        counts = Counter(name for names in df["label_names"] for name in names)
        unlabeled = (df["labels"].str.len() == 0).mean()
        words = df["text"].str.split().str.len()
        print(f"\n== {split}: {len(df):,} sentences | {unlabeled:.1%} fair (no label) "
              f"| median {words.median():.0f} words")
        for name, n in counts.most_common():
            print(f"   {name:<25} {n:>5}")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
