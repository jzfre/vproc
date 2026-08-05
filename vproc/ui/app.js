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

async function boot() {
  state.memories = await (await fetch("/api/memories")).json();
  const picker = $("memory-picker");
  picker.innerHTML = state.memories.map(m =>
    `<option value="${escapeHtml(m.memory_id)}">${escapeHtml(m.memory_id)}</option>`).join("");
  picker.onchange = () => loadMemory(picker.value);
  const wanted = new URLSearchParams(location.search).get("memory");
  if (state.memories.length === 0) { showEmptyState(); return; }
  loadMemory(wanted && state.memories.some(m => m.memory_id === wanted) ? wanted : state.memories[0].memory_id);
}

function showEmptyState() {
  const picker = $("memory-picker");
  picker.innerHTML = `<option value="">no memories</option>`;
  picker.disabled = true;
  $("meeting-meta").textContent = "";
  $("video").hidden = true;
  const fallback = $("video-fallback");
  fallback.hidden = false;
  fallback.innerHTML = `<p>No memories yet.</p>
    <p class="detail">Run <code>vproc ingest &lt;video.mp4&gt;</code>, then reload this page.</p>`;
  setNowSpeaking(null);
  $("transcript").innerHTML = `<p class="empty">Transcript segments will appear here once a meeting is ingested.</p>`;
}

async function loadMemory(id) {
  state.current = id;
  history.replaceState(null, "", `?memory=${encodeURIComponent(id)}`);
  $("memory-picker").value = id;
  state.segments = await (await fetch(`/api/memories/${encodeURIComponent(id)}/segments`)).json();
  const m = state.memories.find(x => x.memory_id === id);
  $("meeting-meta").textContent = `${fmt(m.duration_s)} · ${m.speakers.length} speakers · ${m.segment_count} segments`;
  const video = $("video");
  $("video-fallback").hidden = true; video.hidden = false;
  video.src = `/api/media/${encodeURIComponent(id)}`;
  renderTranscript();
  syncActive();
  // Task 6 hooks in here: renderTimeline();
}

function seek(t) { const v = $("video"); v.currentTime = t; v.play?.()?.catch(() => {}); }

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

boot();
