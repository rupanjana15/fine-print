"""Build the personal project guide PDF from docs/GUIDE.md.

    uv run --with markdown --with pypdf --with reportlab python docs/guide/build.py

Markdown → styled HTML (cover, contents, decision cards) → Chrome headless print-to-PDF → page numbers stamped on.
Output: docs/guide/Fine-Print-Guide.pdf (gitignored: rebuild it from the Markdown, which is the source).
"""

import io
import re
import subprocess
import tempfile
from pathlib import Path

import markdown
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "GUIDE.md"
OUT = HERE / "Fine-Print-Guide.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
:root { --ink: #1d1f24; --muted: #5f6470; --line: #dfe2e8; --soft: #f5f6f9; --accent: #3b5bdb;
        --accent-soft: #eef2ff; --blocker: #c92a2a; --bad: #e8590c; --good: #2f9e44; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; color: var(--ink); font: 10.4pt/1.55 "Iowan Old Style", Charter, Georgia, serif; }
h1, h2, h3, h4, th, .label, .toc, .cover { font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; }
h2 { break-before: page; font-size: 20pt; margin: 0 0 14pt; padding-bottom: 6pt; border-bottom: 2.5pt solid var(--accent);
     letter-spacing: -0.01em; }
h3 { font-size: 13pt; margin: 18pt 0 6pt; color: var(--accent); break-after: avoid; }
h4 { font-size: 11pt; margin: 0 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 8pt; padding-left: 16pt; }
li { margin-bottom: 3pt; }
a { color: var(--accent); text-decoration: none; }
strong { font-weight: 650; }
code { font: 8.6pt/1.4 "SF Mono", Menlo, monospace; background: var(--soft); padding: 0.5pt 3pt; border-radius: 3pt; }
pre { background: var(--soft); border: 0.6pt solid var(--line); border-radius: 5pt; padding: 8pt 10pt; overflow: hidden;
      white-space: pre-wrap; word-break: break-word; break-inside: avoid; margin: 0 0 10pt; }
pre code { background: none; padding: 0; font-size: 7.8pt; }
table { width: 100%; border-collapse: collapse; margin: 4pt 0 12pt; font-size: 8.8pt; line-height: 1.4; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { text-align: left; font-size: 8.2pt; font-weight: 650; color: var(--muted); text-transform: uppercase;
     letter-spacing: 0.03em; border-bottom: 1.2pt solid var(--ink); padding: 4pt 6pt; }
td { border-bottom: 0.5pt solid var(--line); padding: 4pt 6pt; vertical-align: top; }
tbody tr:nth-child(even) td { background: #fafbfc; }
blockquote { margin: 6pt 0 10pt; padding: 7pt 11pt 7pt 12pt; background: var(--accent-soft); border-left: 3pt solid var(--accent);
             border-radius: 0 5pt 5pt 0; break-inside: avoid; }
blockquote::before { content: "Say it"; display: block; font: 650 7.4pt -apple-system, "Helvetica Neue", sans-serif;
                     text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); margin-bottom: 2pt; }
blockquote p { margin: 0; font-style: italic; }
section.card { border: 0.7pt solid var(--line); border-radius: 6pt; padding: 9pt 12pt 3pt; margin: 0 0 9pt;
               break-inside: avoid; }
section.card h4 { color: var(--ink); }
section.card h4 .num { color: var(--accent); margin-right: 4pt; }
hr { display: none; }
h2#part-1-the-project-in-one-page, h2#part-10-whats-next, h2#appendix-repo-map-and-commands { break-before: auto; margin-top: 24pt; }
.cover { height: 252mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; }
.cover .top { border-top: 5pt solid var(--accent); padding-top: 18mm; }
.cover .kicker { font-size: 10pt; letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent); font-weight: 650; }
.cover h1 { font-size: 40pt; line-height: 1.05; margin: 8mm 0 6mm; letter-spacing: -0.02em; }
.cover .sub { font: 13pt/1.5 "Iowan Old Style", Charter, Georgia, serif; color: var(--muted); max-width: 140mm; }
.cover .flags { display: flex; gap: 6pt; margin-top: 12mm; }
.cover .flags span { font-size: 8.5pt; font-weight: 650; color: white; padding: 3pt 8pt; border-radius: 10pt; }
.cover .meta { font-size: 9pt; color: var(--muted); border-top: 0.6pt solid var(--line); padding-top: 4mm; line-height: 1.7; }
.toc { break-after: page; }
.toc h2 { break-before: auto; }
.toc ol { list-style: none; padding: 0; margin: 0; }
.toc > ol > li { font-size: 11pt; font-weight: 650; margin: 7pt 0 2pt; }
.toc ol ol { padding-left: 14pt; }
.toc ol ol li { font-size: 9.2pt; font-weight: 400; margin: 1.5pt 0; color: var(--muted); }
.toc a { color: inherit; }
"""


def cover(md_head: str) -> str:
    sub = re.search(r"^\*(.+?)\*$", md_head, re.MULTILINE)
    ver = re.search(r"^(Version .+)$", md_head, re.MULTILINE)
    return f"""<div class="cover"><div class="top">
      <div class="kicker">Personal project guide</div>
      <h1>Fine Print</h1>
      <div class="sub">{sub.group(1) if sub else ""}</div>
      <div class="flags"><span style="background:#c92a2a">blocker</span><span style="background:#e8590c">bad</span>
        <span style="background:#868e96">neutral</span><span style="background:#2f9e44">good</span></div>
    </div>
    <div class="meta">A fine-tuned 3B language model that flags risky clauses in Terms of Service.<br>
      {ver.group(1) if ver else ""}<br>Decisions D01–D47 · sources: docs/DECISIONS.md, docs/CONCEPTS.md, results/</div></div>"""


def toc_html(tokens: list[dict]) -> str:
    def items(ts, depth):
        if not ts or depth > 2:
            return ""
        lis = "".join(f'<li><a href="#{t["id"]}">{t["name"]}</a>{items(t["children"], depth + 1) if depth < 2 else ""}</li>'
                      for t in ts)
        return f"<ol>{lis}</ol>"
    parts = [t for t in tokens if t["level"] == 2 and not t["name"].startswith("How to use")]
    return f'<div class="toc"><h2 id="contents">Contents</h2>{items(parts, 1)}</div>'


def cards(html: str) -> str:
    """Wrap each h4 (a decision card) and what follows it, up to the next heading, in a <section>."""
    html = re.sub(r"(<h4[^>]*>.*?)(?=<h[234][ >]|$)", r'<section class="card">\1</section>', html, flags=re.DOTALL)
    return re.sub(r'(<h4[^>]*>)(D[\d, →D]+) ·', r'\1<span class="num">\2</span>·', html)


def stamp_page_numbers(pdf_bytes: bytes) -> bytes:
    writer = PdfWriter(clone_from=io.BytesIO(pdf_bytes))
    n = len(writer.pages)
    for i, page in enumerate(writer.pages):
        if i > 0:
            w, h = float(page.mediabox.width), float(page.mediabox.height)
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=(w, h))
            c.setFont("Helvetica", 7.5)
            c.setFillColor(HexColor("#8a8f99"))
            c.drawString(51, 28, "Fine Print · project guide")
            c.drawRightString(w - 51, 28, f"{i + 1} / {n}")
            c.save()
            buf.seek(0)
            page.merge_page(PdfReader(buf).pages[0])
    writer.add_metadata({"/Title": "Fine Print: the whole project, explained", "/Author": "Rupanjana Choudhury"})
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def main() -> None:
    text = SRC.read_text()
    head, body = text.split("\n## ", 1)
    body = "## " + body
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    html = cards(md.convert(body))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Fine Print guide</title>'
            f"<style>{CSS}</style></head><body>{cover(head)}{toc_html(md.toc_tokens)}{html}</body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        src, pdf = Path(tmp) / "guide.html", Path(tmp) / "guide.pdf"
        src.write_text(page)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", src.as_uri()], check=True, capture_output=True, timeout=180)
        OUT.write_bytes(stamp_page_numbers(pdf.read_bytes()))
    print(f"{OUT} ({len(PdfReader(OUT).pages)} pages)")


if __name__ == "__main__":
    main()
