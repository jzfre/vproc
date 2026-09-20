from typing import Literal

from pydantic import BaseModel, Field


def mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


class Project(BaseModel):
    id: str
    name: str
    created_at: str


class Memory(BaseModel):
    id: str
    project_id: str
    title: str
    source_video: str
    duration_s: float | None = None
    status: str = "ready"
    created_at: str | None = None


class Segment(BaseModel):
    id: str
    project_id: str
    memory_id: str
    screen_state_id: str | None = None
    start_ts: float
    end_ts: float
    speaker: str
    said_text: str
    on_screen_text: str = ""
    on_screen_confidence: float | None = None
    frame_path: str | None = None
    source_video: str
    embed_text: str


class Evidence(BaseModel):
    key: str
    segment_id: str
    memory_title: str
    start_ts: float
    end_ts: float
    speaker: str
    text: str
    # Retrieval always returns the k nearest segments, so expose how near (cosine; values are
    # embedding-model specific) and whether the segment literally contains every query word.
    similarity: float | None = None
    keyword_match: bool = False


class Citation(BaseModel):
    memory_title: str
    start_ts: float
    end_ts: float
    speaker: str


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    citations: list[Citation] = Field(default_factory=list)


class Answer(BaseModel):
    answered: bool
    abstained: bool
    text: str
    claims: list[Claim]
    evidence: list[Evidence]
    # Diagnostics for tuning sim_floor / hhem_threshold. Numbers and an enum only: the text of
    # a rejected claim is unverified content and must never be returned.
    abstain_reason: Literal["no_hits", "low_similarity", "no_answer", "unverified"] | None = None
    top_sim: float | None = None
    hhem_scores: list[float] | None = None
