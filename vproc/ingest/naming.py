import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

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


def _canonicalize(all_names: list[str]) -> dict[str, str]:
    """Map each OCR spelling variant to its group's most frequent form."""
    counts = Counter(n.strip() for n in all_names if n and n.strip())
    groups: list[list[str]] = []
    for name in sorted(counts, key=lambda n: -counts[n]):
        for g in groups:
            if SequenceMatcher(None, name.lower(), g[0].lower()).ratio() >= SIMILARITY:
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
    canon = _canonicalize(all_names)
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
