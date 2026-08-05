const state = {memories: [], current: null, segments: []};
const $ = (id) => document.getElementById(id);
const fmt = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

const SPEAKER_COLORS = ["#e6a23c", "#5ec8d8", "#b18cf0", "#6fbf73", "#e57373",
                        "#f0c674", "#8aa9f9", "#d98fc0"];
const speakerColor = new Map();
function colorFor(speaker) {
  if (!speakerColor.has(speaker))
    speakerColor.set(speaker, SPEAKER_COLORS[speakerColor.size % SPEAKER_COLORS.length]);
  return speakerColor.get(speaker);
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => (
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]
  ));
}

// fetch() + res.ok + .json(), collapsed into one throwing call so callers can
// render a friendly inline error instead of crashing on a network failure or
// an error response whose JSON body still parses fine (e.g. a 404 {detail}).
async function fetchJson(url, opts) {
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    throw new Error("network error — is the server running?");
  }
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch (_) { /* body wasn't JSON */ }
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function boot() {
  let memories;
  try {
    memories = await fetchJson("/api/memories");
  } catch (e) {
    showEmptyState(e.message);
    return;
  }
  state.memories = Array.isArray(memories) ? memories : [];
  const picker = $("memory-picker");
  picker.innerHTML = state.memories.map(m =>
    `<option value="${escapeHtml(m.memory_id)}">${escapeHtml(m.memory_id)}</option>`).join("");
  picker.onchange = () => loadMemory(picker.value);
  const wanted = new URLSearchParams(location.search).get("memory");
  if (state.memories.length === 0) { showEmptyState(); return; }
  loadMemory(wanted && state.memories.some(m => m.memory_id === wanted) ? wanted : state.memories[0].memory_id);
}

function showEmptyState(errorMessage) {
  const picker = $("memory-picker");
  picker.innerHTML = `<option value="">no memories</option>`;
  picker.disabled = true;
  $("meeting-meta").textContent = "";
  $("video").hidden = true;
  const fallback = $("video-fallback");
  fallback.hidden = false;
  fallback.innerHTML = errorMessage
    ? `<p>Could not load meetings.</p><p class="detail">${escapeHtml(errorMessage)}</p>`
    : `<p>No meetings ingested yet.</p>
       <p class="detail">Run <code>vproc ingest &lt;video&gt;</code>, then reload this page.</p>`;
  setNowSpeaking(null);
  $("transcript").innerHTML = errorMessage
    ? `<p class="empty">Reload the page once the server is back.</p>`
    : `<p class="empty">Transcript segments will appear here once a meeting is ingested.</p>`;
}

async function loadMemory(id) {
  state.current = id;
  history.replaceState(null, "", `?memory=${encodeURIComponent(id)}`);
  $("memory-picker").value = id;
  let segments;
  try {
    segments = await fetchJson(`/api/memories/${encodeURIComponent(id)}/segments`);
  } catch (e) {
    state.segments = [];
    $("transcript").innerHTML = `<p class="empty">Could not load this meeting's transcript.</p>
      <p class="empty empty-detail">${escapeHtml(e.message)}</p>`;
    return;
  }
  state.segments = Array.isArray(segments) ? segments : [];
  const m = state.memories.find(x => x.memory_id === id);
  $("meeting-meta").textContent = m
    ? `${fmt(m.duration_s)} · ${m.speakers.length} speakers · ${m.segment_count} segments` : "";
  const video = $("video");
  $("video-fallback").hidden = true; video.hidden = false;
  video.src = `/api/media/${encodeURIComponent(id)}`;
  renderTranscript();
  renderTimeline();
  syncActive();
}

function seek(t) { const v = $("video"); v.currentTime = t; v.play?.()?.catch(() => {}); }

// Speaker timeline: one lane per speaker (first-appearance order in the
// transcript), turn rects positioned/sized proportionally to the meeting's
// duration, a shared playhead overlay kept in sync from syncActive(), and
// click-to-seek anywhere in a lane's track.
function renderTimeline() {
  const timeline = $("timeline");
  const m = state.memories.find(x => x.memory_id === state.current);
  const duration = m ? m.duration_s : 0;
  if (!duration || state.segments.length === 0) {
    timeline.innerHTML = `<p class="tl-empty">no segments to plot</p>`;
    return;
  }
  const order = [];
  const bySpeaker = new Map();
  for (const s of state.segments) {
    if (!bySpeaker.has(s.speaker)) { bySpeaker.set(s.speaker, []); order.push(s.speaker); }
    bySpeaker.get(s.speaker).push(s);
  }
  const lanes = order.map((speaker) => {
    const color = colorFor(speaker);
    const turns = bySpeaker.get(speaker).map((s) => {
      const left = (s.start_ts / duration) * 100;
      const width = Math.max(((s.end_ts - s.start_ts) / duration) * 100, 0.4);
      return `<div class="turn" style="left:${left}%;width:${width}%;--c:${color}"
                   title="${fmt(s.start_ts)}–${fmt(s.end_ts)}"></div>`;
    }).join("");
    return `
      <div class="lane">
        <div class="lane-label" style="--c:${color}">${escapeHtml(speaker)}</div>
        <div class="lane-track">${turns}</div>
      </div>`;
  }).join("");
  timeline.innerHTML = `<div class="lanes">${lanes}<div id="playhead"></div></div>`;
  updatePlayhead();
}

// Positions #playhead in pixels, measured off a live .lane-track's
// offsetLeft/offsetWidth (relative to .lanes, its positioned ancestor) so it
// lines up with turn rects exactly regardless of the label column's width.
function updatePlayhead() {
  const playhead = $("playhead");
  const track = document.querySelector("#timeline .lane-track");
  if (!playhead || !track) return;
  const m = state.memories.find(x => x.memory_id === state.current);
  const duration = m ? m.duration_s : 0;
  const frac = duration ? Math.min(Math.max($("video").currentTime / duration, 0), 1) : 0;
  playhead.style.left = `${track.offsetLeft + frac * track.offsetWidth}px`;
}

$("timeline").addEventListener("click", (e) => {
  const track = e.target.closest(".lane-track");
  const m = state.memories.find(x => x.memory_id === state.current);
  if (!track || !m) return;
  const rect = track.getBoundingClientRect();
  const frac = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1);
  seek(frac * m.duration_s);
});

function renderTranscript() {
  const id = state.current;
  const html = state.segments.map((s, i) => {
    const color = colorFor(s.speaker);
    const screen = s.on_screen_text ? `
      <div class="screen">
        <div class="screen-label">on screen</div>
        <pre class="screen-text">${escapeHtml(s.on_screen_text)}</pre>
        ${s.frame_name ? `<img class="frame-thumb" loading="lazy" alt="on-screen capture at ${fmt(s.start_ts)}"
             src="/api/frames/${encodeURIComponent(id)}/${encodeURIComponent(s.frame_name)}">` : ""}
      </div>` : "";
    return `
      <article class="seg" data-index="${i}" data-start="${s.start_ts}" data-end="${s.end_ts}" style="--c:${color}">
        <header class="seg-head" role="button" tabindex="0" aria-label="Seek to ${fmt(s.start_ts)}">
          <span class="seg-dot"></span>
          <span class="seg-speaker">${escapeHtml(s.speaker)}</span>
          <span class="seg-time">${fmt(s.start_ts)}</span>
        </header>
        <p class="seg-text">${escapeHtml(s.said_text || "")}</p>
        ${screen}
      </article>`;
  }).join("");
  $("transcript").innerHTML = html || `<p class="empty">No transcript segments for this meeting.</p>`;
}

function seekFromHeader(target) {
  const head = target.closest(".seg-head");
  if (!head) return;
  const seg = head.closest(".seg");
  seek(Number(seg.dataset.start));
}
$("transcript").addEventListener("click", (e) => seekFromHeader(e.target));
$("transcript").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " ") return;
  if (!e.target.closest(".seg-head")) return;
  e.preventDefault();
  seekFromHeader(e.target);
});

function setNowSpeaking(speaker) {
  const el = $("now-speaking");
  if (speaker) {
    el.style.setProperty("--c", colorFor(speaker));
    el.classList.add("live");
    el.innerHTML = `<span class="dot"></span><span class="label">${escapeHtml(speaker)}</span>`;
  } else {
    el.style.removeProperty("--c");
    el.classList.remove("live");
    el.innerHTML = `<span class="dot"></span><span class="label">silence</span>`;
  }
}

function syncActive() {
  updatePlayhead();
  const t = $("video").currentTime;
  const activeIndex = state.segments.findIndex(s => t >= s.start_ts && t < s.end_ts);
  const hovering = $("transcript").matches(":hover");
  $("transcript").querySelectorAll(".seg").forEach((row) => {
    const isActive = Number(row.dataset.index) === activeIndex;
    row.classList.toggle("active", isActive);
    if (isActive && !hovering) row.scrollIntoView({block: "nearest"});
  });
  setNowSpeaking(activeIndex >= 0 ? state.segments[activeIndex].speaker : null);
}

$("video").addEventListener("timeupdate", syncActive);
$("video").addEventListener("error", async () => {
  const video = $("video");
  const fallback = $("video-fallback");
  video.hidden = true;
  fallback.hidden = false;
  let detail = "";
  if (state.current) {
    try {
      const res = await fetch(`/api/media/${encodeURIComponent(state.current)}`, {headers: {Range: "bytes=0-0"}});
      if (res.status === 404) detail = (await res.json()).detail || "";
    } catch (_) { /* couldn't reach the server to explain why; fall through to the codec note */ }
  }
  fallback.innerHTML = detail
    ? `<p>Video unavailable.</p><p class="detail">${escapeHtml(detail)}</p>`
    : `<p>Video unavailable.</p>
       <p class="detail">Playback failed. If this is an .mkv recording, try Chrome — Safari and
       Firefox don't ship built-in mkv/HEVC support.</p>`;
});

// ---------- ask / search panel ----------

// No fetch timeout here on purpose: `ask` runs a thinking/grounding model and
// can legitimately take well over a minute. The spinner + disabled controls
// are the only "is this still working" signal, not a request deadline.
async function runQuery() {
  const mode = $("qa-mode").value;
  const q = $("qa-input").value.trim();
  if (!q) return;
  $("qa-spinner").hidden = false;
  $("qa-go").disabled = true;
  $("qa-input").disabled = true;
  try {
    if (mode === "ask") {
      const a = await fetchJson("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: q}),
      });
      renderAskResult(a);
    } else {
      const hits = await fetchJson("/search", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query: q}),
      });
      renderSearchResults(hits);
    }
  } catch (e) {
    $("qa-results").innerHTML = `<p class="qa-error">${escapeHtml(e.message)}</p>`;
  } finally {
    $("qa-spinner").hidden = true;
    $("qa-go").disabled = false;
    $("qa-input").disabled = false;
  }
}

function renderAskResult(a) {
  const results = $("qa-results");
  if (a.abstained || !a.answered) {
    results.innerHTML = `<p class="qa-abstain">${escapeHtml(a.text || "Not discussed in these meetings.")}</p>`;
    return;
  }
  const html = a.claims.map((claim) => {
    const chips = claim.citations.map((c) => `
      <button type="button" class="chip" style="--c:${colorFor(c.speaker)}"
              data-memory="${escapeHtml(c.memory_title)}" data-start="${c.start_ts}">${escapeHtml(c.memory_title)} · ${fmt(c.start_ts)} · ${escapeHtml(c.speaker)}</button>`).join("");
    return `
      <div class="claim">
        <p class="claim-text">${escapeHtml(claim.text)}</p>
        <div class="chips">${chips}</div>
      </div>`;
  }).join("");
  results.innerHTML = html || `<p class="qa-abstain">${escapeHtml(a.text || "")}</p>`;
}

function renderSearchResults(hits) {
  const results = $("qa-results");
  if (!Array.isArray(hits) || hits.length === 0) {
    results.innerHTML = `<p class="qa-abstain">No matching segments found.</p>`;
    return;
  }
  results.innerHTML = hits.map((e) => `
    <div class="hit" role="button" tabindex="0" data-memory="${escapeHtml(e.memory_title)}" data-start="${e.start_ts}">
      <div class="hit-head">
        <span class="hit-speaker" style="--c:${colorFor(e.speaker)}">${escapeHtml(e.speaker)}</span>
        <span class="hit-time">${fmt(e.start_ts)}–${fmt(e.end_ts)}</span>
        <span class="hit-memory">${escapeHtml(e.memory_title)}</span>
      </div>
      <p class="hit-text">${escapeHtml(e.text)}</p>
    </div>`).join("");
}

// Citations/hits carry a memory_title that is actually the memory_id (see
// vproc/answer/evidence.py); jump across meetings when it differs from the
// one currently loaded, then seek within it.
async function jumpTo(c) {
  if (c.memory_title !== state.current) {
    await loadMemory(c.memory_title);
  }
  seek(c.start_ts);
}

function jumpFromResults(target) {
  const el = target.closest(".chip, .hit");
  if (!el) return;
  jumpTo({memory_title: el.dataset.memory, start_ts: Number(el.dataset.start)});
}

$("qa-go").addEventListener("click", runQuery);
$("qa-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runQuery();
});
$("qa-mode").addEventListener("change", () => {
  $("qa-input").placeholder = $("qa-mode").value === "ask" ? "Ask this meeting…" : "Search this meeting…";
});
$("qa-results").addEventListener("click", (e) => jumpFromResults(e.target));
$("qa-results").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " ") return;
  if (!e.target.closest(".hit")) return;
  e.preventDefault();
  jumpFromResults(e.target);
});

boot();
