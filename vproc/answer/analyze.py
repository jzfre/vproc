"""Whole-transcript analysis: unlike the grounded, retrieval-based `ask` (which
cites-or-abstains on the top-K matching chunks), this feeds the ENTIRE transcript
of one memory to the grounding model and answers a freeform question about it —
summaries, action items, topics, decisions. No retrieval, no HHEM abstention.
Best for short/medium meetings whose transcript fits the model context."""

from vproc.llm import client

_SYSTEM = (
    "You analyze a single meeting transcript and answer the user's question about it. "
    "Each line is '[<seconds>s <SPEAKER>] text'. Base your answer only on the transcript; "
    "do not invent facts. When you state something, reference the speaker and approximate "
    "timestamp. If the transcript genuinely contains nothing relevant, say so plainly. "
    "Return JSON of the form {\"answer\": string}."
)


def _transcript_text(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: r.get("start_ts", 0.0))
    lines = []
    for r in rows:
        said = (r.get("said_text") or "").strip()
        if said:
            lines.append(f"[{r.get('start_ts', 0.0):.0f}s {r.get('speaker', '?')}] {said}")
    return "\n".join(lines)


def analyze_memory(store, cfg, memory_id: str, question: str,
                   project_id: str = "default") -> str:
    """Load the full transcript for `memory_id` and answer `question` about the whole
    thing. Returns the model's freeform answer string."""
    rows = store.memory_rows(memory_id, project_id)
    transcript = _transcript_text(rows)
    if not transcript:
        return f"No transcript found for memory '{memory_id}'."

    import json

    raw = client.chat_json(
        cfg.grounding.base_url,
        cfg.grounding.model,
        system=_SYSTEM,
        user=f"Transcript:\n{transcript}\n\nQuestion: {question}",
        schema={"type": "object", "properties": {"answer": {"type": "string"}},
                "required": ["answer"]},
        max_tokens=cfg.grounding_max_tokens,
        # Disable the thinking channel: on long inputs it otherwise consumes the
        # whole budget and returns empty content. The schema keeps output valid JSON.
        think=False,
    )
    try:
        return json.loads(raw)["answer"]
    except (ValueError, KeyError, TypeError):
        # Fall back to whatever the model returned rather than crashing the CLI.
        return raw.strip() or "(model returned no answer)"
