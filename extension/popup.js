// Fine Print — popup: server status, start/cancel, progress, result with clickable flags.
const SEVERITIES = ["blocker", "bad", "good", "neutral"];
const START_CMD = "uv run --extra eval python -m fineprint.inference.server";
const main = document.getElementById("main");
let tabId = null;
let online = false;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const send = (msg) => chrome.runtime.sendMessage({ tabId, ...msg });
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

async function checkServer() {
  const r = await send({ type: "health" }).catch(() => ({ error: "unreachable" }));
  online = !r.error;
  document.getElementById("dot").className = `dot ${online ? "on" : "off"}`;
  document.getElementById("server").textContent = online ? (r.loaded ? "model ready" : "server on") : "server off";
}

function renderIdle(error) {
  main.innerHTML = `
    ${error ? `<p class="error">${esc(error)}</p>` : ""}
    <p>Reads this page's terms with a small model running on your computer, then highlights the clauses that matter.</p>
    <div class="row">
      <button class="primary" id="go" ${online ? "" : "disabled"}>Analyse this page</button>
      <button id="sel" ${online ? "" : "disabled"}>Analyse selected text</button>
    </div>
    ${online ? `<p class="small muted">Takes about 1.5–5 s per ~200-word chunk on a laptop CPU; a typical policy, 1–3 minutes.</p>`
             : `<p class="small">Start the local server from the project folder:</p><pre>${START_CMD}</pre>
                <p class="small muted">The first run downloads the 1.9 GB model from Hugging Face.</p>`}
    <p class="note">Fine Print estimate, not legal advice and not <a href="https://tosdr.org" target="_blank">ToS;DR</a>'s grade.</p>`;
  document.getElementById("go").onclick = () => start(false);
  document.getElementById("sel").onclick = () => start(true);
}

async function start(selectionOnly) {
  main.innerHTML = `<p class="muted">Sending the page text to the local model…</p>`;
  await send({ type: "analyse", selectionOnly });
  render(await currentState());
}

function renderProgress(s) {
  const pct = s.total ? Math.round((100 * s.done) / s.total) : 0;
  const elapsed = (Date.now() - s.started) / 1000;
  const left = s.done > 0 ? Math.round(((elapsed / s.done) * (s.total - s.done)) / 60 * 10) / 10 : null;
  main.innerHTML = `
    <p>${s.status === "queued" ? "Waiting for the model (loading it the first time takes ~10 s)…"
                                : `Reading chunk ${Math.min(s.done + 1, s.total)} of ${s.total}`}</p>
    <div class="bar"><div style="width:${pct}%"></div></div>
    <p class="small muted">${left !== null ? `About ${left < 1 ? "<1" : left} min left. ` : ""}You can close this popup; the
      analysis keeps running and the flags appear on the page.</p>
    <button id="cancel">Cancel</button>`;
  document.getElementById("cancel").onclick = () => send({ type: "cancel" });
}

function renderResult(s) {
  const r = s.result;
  const found = s.found || [];
  const flags = r.flags.map((f, i) => ({ ...f, i, onPage: Boolean(found[i]) }))
    .sort((a, b) => SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity));
  const item = (f) => `
    <li style="--c:var(--${esc(f.severity || "neutral")})" class="${f.onPage ? "" : "missing"}" data-i="${f.i}">
      <div class="cat">${esc(f.severity)} · ${esc((f.category || "").replaceAll("_", " "))}</div>
      <div>${esc(f.explanation)}</div>
      ${f.onPage ? "" : `<div class="where">quote not found on the page</div>`}
    </li>`;
  const main_ = flags.filter((f) => f.severity !== "neutral");
  const neutral = flags.filter((f) => f.severity === "neutral");
  main.innerHTML = `
    <div class="result-head">
      <div class="grade" style="background:var(--${esc(r.grade)})">${esc(r.grade)}</div>
      <div>
        <div><b>Fine Print estimate</b> · ${plural(r.flags.length, "flag")} in ${plural(r.n_chunks, "chunk")}, ${Math.round(r.seconds)} s</div>
        <div class="counts">${SEVERITIES.map((k) => `<span style="--c:var(--${k})">${r.counts[k] || 0} ${k}</span>`).join("")}</div>
      </div>
    </div>
    ${s.selection ? `<p class="small muted">Selected text only. The grade is for this text, not the whole service.</p>` : ""}
    <ul>${main_.map(item).join("") || `<p class="muted">Nothing good or bad flagged.</p>`}</ul>
    ${neutral.length ? `<details><summary>${plural(neutral.length, "neutral flag")}</summary><ul>${neutral.map(item).join("")}</ul></details>` : ""}
    <div class="row" style="margin-top:10px"><button id="again">Analyse again</button></div>
    <p class="note">Not legal advice. On whole services the grade matches ToS;DR's for about half of companies and is
      within one grade for ~9 in 10; it tends to be a grade too kind on the worst ones. Click a flag to see its quote.</p>`;
  main.querySelectorAll("li:not(.missing)").forEach((li) => {
    li.onclick = () => send({ type: "scrollTo", index: Number(li.dataset.i) });
  });
  document.getElementById("again").onclick = async () => { await send({ type: "reset" }); renderIdle(); };
}

function render(s) {
  if (!s || s.status === "cancelled") return renderIdle();
  if (s.status === "error") return renderIdle(s.error);
  if (s.status === "queued" || s.status === "running") return renderProgress(s);
  if (s.status === "done") return renderResult(s);
  renderIdle();
}

async function currentState() {
  return (await chrome.storage.session.get(`tab:${tabId}`))[`tab:${tabId}`] || null;
}

chrome.storage.session.onChanged.addListener((changes) => {
  const c = changes[`tab:${tabId}`];
  if (c) render(c.newValue);
});

(async () => {
  [{ id: tabId }] = await chrome.tabs.query({ active: true, currentWindow: true });
  await checkServer();
  const s = await currentState();
  if (s && (s.status === "queued" || s.status === "running")) send({ type: "resume" });
  render(s);
})();
