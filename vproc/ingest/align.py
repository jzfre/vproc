import uuid
from dataclasses import dataclass, field

from vproc.ingest.frames import RawFrame
from vproc.ingest.transcribe import TranscriptSegment
from vproc.models import Segment, mmss


@dataclass
class ScreenState:
    screen_state_id: str
    t_start: float
    t_end: float
    frame_path: str
    on_screen_text: str = ""


def build_screen_states(frames: list[RawFrame], end_time: float) -> list[ScreenState]:
    states: list[ScreenState] = []
    for i, f in enumerate(frames):
        t_end = frames[i + 1].t if i + 1 < len(frames) else end_time
        states.append(ScreenState(f"ss{i}", f.t, t_end, f.path))
    return states


def _midpoint(seg: TranscriptSegment) -> float:
    return (seg.start + seg.end) / 2.0


def assign_segments(states: list[ScreenState],
                    transcript: list[TranscriptSegment]) -> dict[str, list[TranscriptSegment]]:
    buckets: dict[str, list[TranscriptSegment]] = {s.screen_state_id: [] for s in states}
    for seg in transcript:
        mid = _midpoint(seg)
        chosen = next((s for s in states if s.t_start <= mid < s.t_end), None)
        if chosen is None and states:
            chosen = states[-1] if mid >= states[-1].t_start else states[0]
        if chosen is not None:
            buckets[chosen.screen_state_id].append(seg)
    return buckets


def make_embed_text(on_screen: str, segs: list[TranscriptSegment]) -> str:
    spoken = "\n".join(f"{s.speaker} ({mmss(s.start)}): {s.text}" for s in segs)
    return f"[SCREEN]\n{on_screen}\n[SPOKEN]\n{spoken}"


# Sub-split long monologues so one near-static screen doesn't collapse into a single
# un-citable blob. Each emitted segment spans at most MAX_WINDOW_S seconds.
MAX_WINDOW_S = 90.0


def _subsplit(segs: list[TranscriptSegment], max_window: float) -> list[list[TranscriptSegment]]:
    """Split an ordered transcript list into chunks, breaking when adding the next
    segment would exceed `max_window` seconds OR when the speaker changes."""
    chunks: list[list[TranscriptSegment]] = []
    cur: list[TranscriptSegment] = []
    for seg in segs:
        if cur and (seg.end - cur[0].start > max_window or seg.speaker != cur[-1].speaker):
            chunks.append(cur)
            cur = []
        cur.append(seg)
    if cur:
        chunks.append(cur)
    return chunks


def _make_segment(project_id: str, memory_id: str, source_video: str,
                  s: ScreenState, segs: list[TranscriptSegment]) -> Segment:
    said = " ".join(x.text for x in segs)
    start = segs[0].start if segs else s.t_start
    end = segs[-1].end if segs else s.t_end
    speaker = segs[0].speaker if segs else "SPEAKER_0"
    return Segment(
        id=str(uuid.uuid4()),
        project_id=project_id, memory_id=memory_id, screen_state_id=s.screen_state_id,
        start_ts=start, end_ts=end, speaker=speaker,
        said_text=said, on_screen_text=s.on_screen_text, on_screen_confidence=None,
        frame_path=s.frame_path, source_video=source_video,
        embed_text=make_embed_text(s.on_screen_text, segs),
    )


def build_segments(project_id: str, memory_id: str, source_video: str,
                   states: list[ScreenState],
                   transcript: list[TranscriptSegment],
                   max_window: float = MAX_WINDOW_S) -> list[Segment]:
    buckets = assign_segments(states, transcript)
    out: list[Segment] = []
    for s in states:
        segs = buckets[s.screen_state_id]
        if not segs:
            # Silent screen with on-screen text still yields one OCR-only segment.
            if s.on_screen_text.strip():
                out.append(_make_segment(project_id, memory_id, source_video, s, []))
            continue
        for chunk in _subsplit(segs, max_window):
            out.append(_make_segment(project_id, memory_id, source_video, s, chunk))
    return out
