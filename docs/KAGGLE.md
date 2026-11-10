# Running the heavy jobs on Kaggle

The Mac (M1, 8 GB) can't run 3B-model inference or DeBERTa training at a useful speed, so these run on
Kaggle's free GPUs (decision D35). One notebook uses **both T4 GPUs at once**:

| GPU | Jobs, in order | Rough time |
|---|---|---|
| 0 | B1 Qwen2.5-3B zero-shot on `test-500`, then `lexglue` | ~1–2 h |
| 1 | B3 DeBERTa-v3-small train + eval, then B2 Qwen 3-shot on `test-500`, then `lexglue` | ~2–3 h |

Times are estimates; the first run will tell.

## One-time account setup
1. Sign in at kaggle.com.
2. *Settings → Phone verification*: needed before Kaggle allows GPUs and internet in notebooks.

## Each run
1. **Make the bundle on the Mac** (only after the dataset or code changed):
   `uv run python -m fineprint.kaggle_bundle` → `dist/kaggle/fine-print-bundle.zip` (~16 MB)
2. **Upload it:** *Datasets → New dataset* → drop the zip → title e.g. `fine-print-bundle` → **Private** → Create.
   (Later runs: open the dataset → *New version* → upload the new zip.)
3. **Import the notebook:** *Code → New notebook → File → Import notebook* → `notebooks/kaggle_baselines.ipynb`.
4. **Notebook settings** (right panel):
   - *Add input* → your `fine-print-bundle` dataset
   - *Accelerator* → **GPU T4 x2**
   - *Internet* → **on** (downloads Qwen and DeBERTa from Hugging Face)
5. **Run in the background:** *Save Version → Save & Run All (Commit) → Save*. You can close the tab.
6. **Collect results:** when the version shows *Complete*, open it → *Output* → download
   `fine-print-results.zip` → put it in the repo root.
7. **Import on the Mac:** `uv run python -m fineprint.kaggle_results`
   This re-scores everything locally, checks Kaggle used the same eval sets, and updates `results/RESULTS.md`.

## If something goes wrong
- The notebook's log cell prints the last line from each GPU every 2 minutes; full logs are in
  `gpu0.log` / `gpu1.log` in the output and inside the results zip.
- Jobs on one GPU run one after another, and a failed job doesn't stop the next one, so partial results
  still come back.
- Paste the error into the Claude Code session and it gets fixed in the repo; then rebuild the bundle
  (step 1) and upload a new version (step 2).

## Privacy and licences
The bundle holds our code and ToS;DR-derived text (CC BY-SA). Keep the Kaggle dataset **private**; we don't
redistribute the raw data. No tokens or secrets are needed.
