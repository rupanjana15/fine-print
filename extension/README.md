# Fine Print browser extension

Click the icon on a Terms of Service or privacy policy page. The page text goes to the Fine Print model **running on
your own computer**, and the flagged clauses are highlighted on the page by severity, with a grade and a list in the
popup. Nothing leaves your machine. Decision D46.

## Install (Chrome, Edge, Brave)

1. Start the local server from the project folder (the first run downloads the 1.9 GB model from Hugging Face):

```bash
uv run --extra eval python -m fineprint.inference.server
```

If you already have the model file, point to it instead of downloading it again:

```bash
FINEPRINT_MODEL=data/models/fine-print-v2-q4_k_m.gguf uv run --extra eval python -m fineprint.inference.server
```

2. Open `chrome://extensions`, switch on **Developer mode**, click **Load unpacked**, and pick this `extension/`
   folder.
3. Open a terms page, click the Fine Print icon, then **Analyse this page** or **Analyse selected text**.

A typical policy takes 1–3 minutes on a laptop CPU (about 1.5 s per chunk with nothing to flag, 5 s per flagged
chunk). You can close the popup while it runs; the highlights appear when it's done. Click a flag in the popup to
scroll to its quote.

## How it works

| Part | What it does |
|---|---|
| `background.js` | Reads the page text (the selection, else the largest `main`/`article`, else the body), sends it to `http://127.0.0.1:8737/analyse`, polls the job, keeps the state per tab |
| `content.js` + `content.css` | Finds each quote on the page (whitespace and case insensitive, across inline tags like `<em>`) and wraps it in a coloured `<mark>` |
| `popup.html` + `popup.js` | Server status, start / cancel, progress with time left, grade and flags |
| `../src/fineprint/inference/server.py` | Local server: the same pipeline as the CLI and the app (chunk → model with threshold → validate → merge → grade) |

The server only answers requests from a browser extension (or curl), so a web page you visit can't use it.

## Testing without installing

`dev/page.html` (highlighting) and `dev/popup.html?state=off|idle|running|done` (popup) run the real scripts with
the Chrome APIs stubbed:

```bash
python3 -m http.server 8766 -d extension
```

Then open `http://localhost:8766/dev/page.html`.

## Limits

- Grades are a "Fine Print estimate", not ToS;DR's grade or legal advice; on whole services the grade is exact for
  about half of companies and within one grade for ~9 in 10 (D43). A single page is less reliable than a whole service.
- Pages longer than 150,000 characters are refused; select one section instead.
- Pages that redraw their text (some single-page apps) can drop the highlights; clicking a flag draws them again.
