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
