"""Pack code + data into one zip to upload as a private Kaggle dataset (decision D35).

The Kaggle notebook (notebooks/kaggle_baselines.ipynb) copies it to /kaggle/working and runs the
heavy jobs there, so the Mac only does light work. Contents: src/, configs/, the processed splits and
the LexGLUE test split. No secrets, no model weights (Kaggle downloads those from the HF Hub).

Usage:  uv run python -m fineprint.kaggle_bundle   → dist/kaggle/fine-print-bundle.zip
"""

import zipfile

from fineprint.config import ROOT, data_path

OUT = ROOT / "dist" / "kaggle" / "fine-print-bundle.zip"


def files():
    yield from (ROOT / "src").rglob("*.py")
    yield from (ROOT / "configs").glob("*.yaml")
    yield from data_path("processed").glob("*.jsonl")
    yield data_path("raw") / "lexglue" / "test.parquet"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files():
            if "__pycache__" not in f.parts:
                z.write(f, f"fine-print/{f.relative_to(ROOT)}")
    print(f"{OUT}  {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
