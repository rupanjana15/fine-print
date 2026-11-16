// Fine Print — service worker. Sends page text to the local server (src/fineprint/inference/server.py),
// polls the job, keeps per-tab state in chrome.storage.session, and asks the content script to highlight quotes.
const SERVER = "http://127.0.0.1:8737";
const POLL_MS = 1500;
const polling = new Set();

const key = (tabId) => `tab:${tabId}`;
async function getState(tabId) {
  return (await chrome.storage.session.get(key(tabId)))[key(tabId)] || null;
}
async function setState(tabId, state) {
  await chrome.storage.session.set({ [key(tabId)]: state });
  return state;
}

async function api(path, options = {}) {
  const r = await fetch(SERVER + path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `server answered ${r.status}`);
  return body;
}

// Runs in the page: the selection, or the main text block, or the whole body.
function extractText(selectionOnly) {
  const sel = String(window.getSelection() || "").trim();
  if (selectionOnly) return { text: sel, url: location.href, title: document.title, selection: true };
  const candidates = [...document.querySelectorAll("main, article, [role=main]")]
    .map((el) => el.innerText || "")
    .filter((t) => t.length > 1500)
    .sort((a, b) => b.length - a.length);
  const text = candidates[0] || document.body.innerText || "";
  return { text, url: location.href, title: document.title, selection: false };
}

async function analyse(tabId, selectionOnly) {
  const [{ result: page }] = await chrome.scripting.executeScript({
    target: { tabId }, func: extractText, args: [selectionOnly],
  });
  if (!page.text.trim()) {
    return setState(tabId, { status: "error", error: selectionOnly ? "Select some text first." : "No text found on this page." });
  }
  await clearHighlights(tabId);
  const job = await api("/analyse", { method: "POST", body: JSON.stringify({ text: page.text, url: page.url }) });
  await setState(tabId, { status: "queued", job: job.job, done: 0, total: job.n_chunks, url: page.url,
                          title: page.title, selection: page.selection, started: Date.now() });
  poll(tabId);
}

async function poll(tabId) {
  if (polling.has(tabId)) return;
  polling.add(tabId);
  try {
    for (;;) {
      const state = await getState(tabId);
      if (!state || !["queued", "running"].includes(state.status)) return;
      let job;
      try {
        job = await api(`/jobs/${state.job}`);
      } catch (e) {
        await setState(tabId, { ...state, status: "error", error: `Lost the local server: ${e.message}` });
        return;
      }
      const next = { ...state, status: job.status, done: job.done, total: job.total };
      if (job.status === "done") {
        next.result = job.result;
        next.found = await highlight(tabId, job.result.flags);
      } else if (job.status === "error") {
        next.error = job.error;
      }
      await setState(tabId, next);
      if (job.status !== "queued" && job.status !== "running") return;
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
  } finally {
    polling.delete(tabId);
  }
}

async function ensureContentScript(tabId) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  await chrome.scripting.insertCSS({ target: { tabId }, files: ["content.css"] });
}

async function highlight(tabId, flags) {
  try {
    await ensureContentScript(tabId);
    return await chrome.tabs.sendMessage(tabId, { type: "highlight", flags });
  } catch {
    return []; // page navigated away or can't be scripted; the popup still lists the flags
  }
}

async function clearHighlights(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "clear" });
  } catch { /* no content script yet */ }
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  (async () => {
    try {
      if (msg.type === "health") return reply(await api("/health"));
      if (msg.type === "analyse") { await analyse(msg.tabId, msg.selectionOnly); return reply({ ok: true }); }
      if (msg.type === "resume") { poll(msg.tabId); return reply({ ok: true }); }
      if (msg.type === "cancel") {
        const state = await getState(msg.tabId);
        if (state?.job) await api(`/jobs/${state.job}`, { method: "DELETE" }).catch(() => {});
        await setState(msg.tabId, { ...state, status: "cancelled" });
        return reply({ ok: true });
      }
      if (msg.type === "reset") {
        await clearHighlights(msg.tabId);
        await chrome.storage.session.remove(key(msg.tabId));
        return reply({ ok: true });
      }
      if (msg.type === "scrollTo") {
        await ensureContentScript(msg.tabId);
        const state = await getState(msg.tabId);
        // Highlights vanish if the page re-rendered; draw them again before scrolling.
        const ok = await chrome.tabs.sendMessage(msg.tabId, { type: "scrollTo", index: msg.index });
        if (!ok && state?.result) {
          await chrome.tabs.sendMessage(msg.tabId, { type: "highlight", flags: state.result.flags });
          await chrome.tabs.sendMessage(msg.tabId, { type: "scrollTo", index: msg.index });
        }
        return reply({ ok: true });
      }
      reply({ error: "unknown message" });
    } catch (e) {
      if (msg.type === "analyse") await setState(msg.tabId, { status: "error", error: e.message });
      reply({ error: e.message });
    }
  })();
  return true; // reply asynchronously
});

chrome.tabs.onRemoved.addListener((tabId) => chrome.storage.session.remove(key(tabId)));
chrome.tabs.onUpdated.addListener(async (tabId, info) => {
  if (!info.url) return;
  const state = await getState(tabId);
  if (state?.url && state.url.split("#")[0] !== info.url.split("#")[0] && state.status !== "running" && state.status !== "queued") {
    await chrome.storage.session.remove(key(tabId)); // a different page: old results don't apply
  }
});
