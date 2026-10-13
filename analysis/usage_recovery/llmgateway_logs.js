// Download LLM Gateway request logs (metadata only) as JSON.
//
// The logs API (internal.llmgateway.io/logs) needs a dashboard login session, not an API key.
// Run this in the browser console on an llmgateway.io/dashboard page while logged in; it saves
// llmgateway-logs.json to the downloads folder; gzip it into results/recovered-usage/records/.
// With metadata-only retention the logs hold no prompts or answers; match.py ties rows to questions.

(async () => {
  const START = "2026-10-09T15:00:00Z", END = "2026-10-13T09:10:00Z";  // main runs, before the re-run
  const MODELS = new Set(["claude-haiku-5-5", "gpt-6-luna", "gemini-3.8-flash", "claude-sonnet-5-5", "gpt-6.1-sol"]);
  const KEEP = ["id", "requestId", "createdAt", "duration", "timeToFirstToken", "timeToFirstReasoningToken",
    "requestedModel", "usedModel", "usedProvider", "responseSize", "finishReason", "unifiedFinishReason",
    "promptTokens", "completionTokens", "reasoningTokens", "cachedTokens", "cost", "hasError", "canceled",
    "streamed", "reasoningEffort", "maxTokens"];
  const rows = [], seen = new Set();
  let cursor = null, fails = 0;
  while (true) {
    const q = new URLSearchParams({limit: "100", startDate: START, endDate: END, orderBy: "createdAt_asc"});
    if (cursor) q.set("cursor", cursor);
    let body;
    try {
      const r = await fetch("https://internal.llmgateway.io/logs?" + q, {credentials: "include"});
      if (r.status === 429 || r.status >= 500) throw new Error("HTTP " + r.status);
      if (!r.ok) throw new Error("stop: HTTP " + r.status);
      body = await r.json(); fails = 0;
    } catch (e) {
      if (String(e).includes("stop") || ++fails > 8) { console.error(e); break; }
      await new Promise(s => setTimeout(s, 2000 * fails)); continue;
    }
    for (const l of body.logs) {
      if (seen.has(l.id) || !MODELS.has(l.requestedModel)) continue;
      seen.add(l.id);
      rows.push(Object.fromEntries(KEEP.map(k => [k, l[k] ?? null])));
    }
    if (!body.pagination.hasMore) break;
    cursor = body.pagination.nextCursor;
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(rows)], {type: "application/json"}));
  a.download = "llmgateway-logs.json";
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`${rows.length} rows saved`);
})();
