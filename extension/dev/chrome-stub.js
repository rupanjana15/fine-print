// Stand-in for the Chrome extension APIs so popup.html can be viewed in a normal tab (?state=off|idle|running|done).
(() => {
  const state = new URLSearchParams(location.search).get("state") || "idle";
  const result = { grade: "D", score: -2, counts: { good: 1, neutral: 1, bad: 2, blocker: 1 }, n_chunks: 31, chars: 18200, seconds: 96,
    flags: [
      { category: "dispute_resolution", severity: "bad", explanation: "You waive your right to a class action", quote: "q" },
      { category: "changes_to_terms", severity: "bad", explanation: "Instead of asking directly, this Service will assume your consent to changes of terms merely from your continuing to use it", quote: "q" },
      { category: "content_license", severity: "blocker", explanation: "The service can use your content for any purpose, including commercially", quote: "q" },
      { category: "content_ownership", severity: "good", explanation: "You keep ownership of your content", quote: "q" },
      { category: "transparency", severity: "neutral", explanation: "Information is provided about what data is collected", quote: "q" },
    ] };
  const states = {
    running: { status: "running", done: 12, total: 31, started: Date.now() - 40000 },
    done: { status: "done", result, found: [true, true, true, false, true] },
  };
  const many = Array.from({ length: 16 }, (_, i) => ({ ...result.flags[i % 5], explanation: result.flags[i % 5].explanation + (i > 4 ? " (repeat)" : "") }));
  states.long = { status: "done", result: { ...result, flags: many }, found: many.map(() => true) };
  const store = { "tab:1": states[state] || null };
  window.chrome = {
    tabs: { query: async () => [{ id: 1 }] },
    runtime: { sendMessage: async (m) => (m.type === "health" ? (state === "off" ? { error: "x" } : { ok: true, loaded: true }) : { ok: true }) },
    storage: { session: { get: async (k) => ({ [k]: store[k] }), onChanged: { addListener() {} } } },
  };
})();
