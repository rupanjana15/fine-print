// Fine Print — content script: finds each flag's quote on the page and highlights it (whitespace-insensitive,
// across element boundaries). Injected on demand by background.js; safe to inject twice.
(() => {
  if (window.__fineprint) return;
  window.__fineprint = true;
  const SKIP = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA", "INPUT", "SELECT"]);

  // Page text with whitespace collapsed (like the server's clean_text), plus a map back to (text node, offset).
  function pageIndex() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (SKIP.has(n.parentElement?.tagName) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    let text = "";
    const map = []; // map[i] = [node, offset] for text[i]
    let lastSpace = true;
    let lastBlock = null;
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const s = node.nodeValue;
      const block = blockOf(node.parentElement);
      if (block !== lastBlock && !lastSpace && map.length) { // new paragraph/cell: words must not run together
        text += " "; map.push([node, 0]); lastSpace = true;
        map[map.length - 1].boundary = true;
      }
      lastBlock = block;
      for (let i = 0; i < s.length; i++) {
        if (/\s/.test(s[i])) {
          if (lastSpace) continue;
          text += " "; map.push([node, i]); lastSpace = true;
        } else {
          text += s[i]; map.push([node, i]); lastSpace = false;
        }
      }
    }
    return { text, map };
  }

  // Nearest ancestor that is laid out as a block (inline elements like <em> or <a> continue the sentence).
  const blocks = new WeakMap();
  function blockOf(el) {
    let e = el;
    while (e && e !== document.body) {
      let b = blocks.get(e);
      if (b === undefined) {
        b = !getComputedStyle(e).display.startsWith("inline");
        blocks.set(e, b);
      }
      if (b) return e;
      e = e.parentElement;
    }
    return document.body;
  }

  const norm = (q) => q.replace(/\s+/g, " ").trim();

  function find(index, quote) {
    const q = norm(quote);
    if (q.length < 8) return -1;
    let at = index.text.indexOf(q);
    if (at < 0) at = index.text.toLowerCase().indexOf(q.toLowerCase());
    return at < 0 ? -1 : at;
  }

  // Wrap text[start, end) in <mark>s, one per text node it touches.
  function wrap(index, start, end, flag, i) {
    const pieces = new Map();
    for (let k = start; k < end; k++) {
      const entry = index.map[k];
      if (entry.boundary) continue; // synthetic space between blocks, not page text
      const [node, off] = entry;
      const p = pieces.get(node) || { from: off, to: off };
      p.from = Math.min(p.from, off); p.to = Math.max(p.to, off + 1);
      pieces.set(node, p);
    }
    let first = null;
    for (const [node, { from, to }] of pieces) {
      const range = document.createRange();
      range.setStart(node, from);
      range.setEnd(node, to);
      const mark = document.createElement("mark");
      mark.className = `fineprint-mark fineprint-${flag.severity || "neutral"}`;
      mark.dataset.fpIndex = String(i);
      mark.title = `${flag.severity || ""} · ${flag.category}: ${flag.explanation}`;
      try {
        range.surroundContents(mark);
        first = first || mark;
      } catch { /* range crosses an element boundary inside one node: skip that piece */ }
    }
    return first !== null;
  }

  function clear() {
    for (const m of document.querySelectorAll("mark.fineprint-mark")) {
      m.replaceWith(...m.childNodes);
    }
    document.body.normalize();
  }

  chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
    if (msg.type === "clear") { clear(); reply(true); }
    if (msg.type === "highlight") {
      clear();
      const found = [];
      // Wrap one at a time, re-indexing after each: wrapping splits text nodes.
      msg.flags.forEach((flag, i) => {
        const index = pageIndex();
        const at = flag.quote ? find(index, flag.quote) : -1;
        found.push(at >= 0 && wrap(index, at, at + norm(flag.quote).length, flag, i));
      });
      reply(found);
    }
    if (msg.type === "scrollTo") {
      const mark = document.querySelector(`mark.fineprint-mark[data-fp-index="${msg.index}"]`);
      if (mark) {
        mark.scrollIntoView({ behavior: "smooth", block: "center" });
        document.querySelectorAll(`mark[data-fp-index="${msg.index}"]`).forEach((m) => {
          m.classList.remove("fineprint-flash");
          void m.offsetWidth; // restart the animation
          m.classList.add("fineprint-flash");
        });
      }
      reply(Boolean(mark));
    }
  });
})();
