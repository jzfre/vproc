import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations

from vproc.ingest.diarize import SpeakerTurn
from vproc.ingest.transcribe import TranscriptSegment
from vproc.llm import client

PROMPT = (
    "This is a frame from a recorded video meeting. Look at the participant tiles / filmstrip "
    "and any visual speaking indicators (highlighted tile border, microphone icon, active-"
    "speaker name label). Answer in strict JSON: "
    '{"speaking": "<name of the participant currently speaking, or null>", '
    '"visible_names": ["<all participant names you can read>"]}'
)

MAX_PROBES_PER_SPEAKER = 6
SIMILARITY = 0.75  # difflib ratio: OCR jitter of one name groups; distinct names don't


@dataclass
class NameVote:
    speaker: str            # diarized cluster label, e.g. "SPEAKER_04"
    t: float                # video timestamp probed
    name: str | None        # active speaker as read; None = not determinable
    visible_names: list[str]


def _canonicalize(all_names: list[str],
                  cooccur: set[frozenset[str]] = frozenset()) -> dict[str, str]:
    """Map each OCR spelling variant to its group's most frequent form. Two spellings that
    co-occurred in one frame's visible_names are confirmed distinct people and never share
    a group, however similar (e.g. "Patino, Daniel" vs "Patino, Daniela")."""
    counts = Counter(n.strip() for n in all_names if n and n.strip())
    groups: list[list[str]] = []
    for name in sorted(counts, key=lambda n: -counts[n]):
        for g in groups:
            similar = SequenceMatcher(None, name.lower(), g[0].lower()).ratio() >= SIMILARITY
            conflicts = any(frozenset((name, m)) in cooccur for m in g)
            if similar and not conflicts:
                g.append(name)
                break
        else:
            groups.append([name])
    canon: dict[str, str] = {}
    for g in groups:
        best = max(g, key=lambda n: counts[n])
        for n in g:
            canon[n] = best
    return canon


def resolve_names(votes: list[NameVote]) -> tuple[dict[str, str], dict]:
    """(mapping, suggestions). A cluster maps to its top canonical name iff that name has
    >= 2 votes AND a strict majority of the cluster's non-null votes; clusters sharing a
    winner both map (diarization over-splits). The rest become evidence-backed suggestions."""
    all_names = [v.name for v in votes if v.name] + [n for v in votes for n in v.visible_names]
    cooccur = {frozenset((a.strip(), b.strip()))
              for v in votes
              for a, b in combinations(v.visible_names, 2)
              if a.strip() and b.strip() and a.strip() != b.strip()}
    canon = _canonicalize(all_names, cooccur)
    by_speaker: dict[str, list[NameVote]] = defaultdict(list)
    for v in votes:
        by_speaker[v.speaker].append(v)
    mapping: dict[str, str] = {}
    suggestions: dict = {}
    for speaker, vs in sorted(by_speaker.items()):
        tally = Counter(canon.get(v.name.strip(), v.name.strip()) for v in vs if v.name and v.name.strip())
        total = sum(tally.values())
        if tally:
            top, n = tally.most_common(1)[0]
            if n >= 2 and n * 2 > total:
                mapping[speaker] = top
                continue
        suggestions[speaker] = {
            "votes": dict(tally),
            "evidence": [{"t": round(v.t), "name": canon.get(v.name.strip(), v.name.strip())}
                         for v in vs if v.name and v.name.strip()],
        }
    return mapping, suggestions


def apply_names(transcript: list[TranscriptSegment], mapping: dict[str, str]) -> None:
    for seg in transcript:
        if seg.speaker in mapping:
            seg.speaker = mapping[seg.speaker]


def _frame_at(video_path: str, t: float, out_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-ss", str(t), "-i", video_path, "-frames:v", "1", out_path, "-y"],
        check=True)


def _parse_vote(raw: str) -> tuple[str | None, list[str]]:
    """Lenient parse of a probe reply; a malformed reply is a lost vote, not an error."""
    try:
        i, j = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[i:j + 1])
        name = data.get("speaking")
        names = data.get("visible_names")
        return (name if isinstance(name, str) and name.strip() else None,
                [n for n in names if isinstance(n, str)] if isinstance(names, list) else [])
    except Exception:
        return None, []


def sample_name_votes(video_path: str, turns: list[SpeakerTurn], cfg, probe=None) -> list[NameVote]:
    """Probe frames at the midpoints of each cluster's longest turns and ask the vision
    model who is speaking. `probe` is the injection seam: (image_path) -> raw reply."""
    if probe is None:
        def probe(path):
            return client.ocr_image(cfg.ocr.base_url, cfg.ocr.model, path, PROMPT,
                                    timeout=cfg.ocr_timeout, max_tokens=cfg.ocr_max_tokens)
    by_speaker: dict[str, list[SpeakerTurn]] = defaultdict(list)
    for t in turns:
        by_speaker[t.speaker].append(t)
    votes: list[NameVote] = []
    attempted = failed = 0
    with tempfile.TemporaryDirectory(prefix="vproc-name-") as work:
        for speaker, spk_turns in sorted(by_speaker.items()):
            longest = sorted(spk_turns, key=lambda t: t.end - t.start, reverse=True)
            for turn in longest[:MAX_PROBES_PER_SPEAKER]:
                mid = (turn.start + turn.end) / 2.0
                frame = os.path.join(work, f"{speaker}-{int(mid * 1000)}.png")
                attempted += 1
                try:
                    _frame_at(video_path, mid, frame)
                    name, visible = _parse_vote(probe(frame))
                except Exception:
                    failed += 1
                    continue  # a failed probe is a lost vote, never a failed ingest
                votes.append(NameVote(speaker, mid, name, visible))
    if failed:
        print(f"warning: {failed}/{attempted} name probes failed", file=sys.stderr)
    return votes


def save_speaker_map(mem_dir: str, data: dict) -> None:
    os.makedirs(mem_dir, exist_ok=True)
    with open(os.path.join(mem_dir, "speakers.json"), "w") as f:
        json.dump(data, f, indent=1)


def load_speaker_map(mem_dir: str) -> dict:
    with open(os.path.join(mem_dir, "speakers.json")) as f:
        return json.load(f)
