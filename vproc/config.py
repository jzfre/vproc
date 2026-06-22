import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str


# Default ASR model for local in-process transcription (mlx-whisper on Apple Silicon).
DEFAULT_TRANSCRIBE_MODEL = "mlx-community/whisper-large-v3-turbo"


@dataclass(frozen=True)
class Config:
    ocr: Endpoint
    embed: Endpoint
    grounding: Endpoint
    index_path: str
    host: str
    port: int
    sim_floor: float
    hhem_threshold: float
    hf_token: str | None
    # Transcription is configurable like the other functions: an empty base_url means
    # local in-process whisper; a base_url means an OpenAI-compatible ASR endpoint.
    transcribe: Endpoint = Endpoint("", DEFAULT_TRANSCRIBE_MODEL)


def _ep(prefix: str, default_url: str, default_model: str) -> Endpoint:
    return Endpoint(
        base_url=os.environ.get(f"{prefix}_BASE_URL", default_url),
        model=os.environ.get(f"{prefix}_MODEL", default_model),
    )


def load_config() -> Config:
    return Config(
        ocr=_ep("VPROC_OCR", "http://voyage:8000/v1", "QuantTrio/Qwen3.5-9B-AWQ"),
        embed=_ep("VPROC_EMBED", "http://localhost:1234/v1", "Qwen3-Embedding-0.6B"),
        grounding=_ep("VPROC_GROUNDING", "http://localhost:1234/v1", "Qwen3-32B-AWQ"),
        transcribe=_ep("VPROC_TRANSCRIBE", "", DEFAULT_TRANSCRIBE_MODEL),
        index_path=os.environ.get("VPROC_INDEX_PATH", "./vproc.lance"),
        host=os.environ.get("VPROC_HOST", "0.0.0.0"),
        port=int(os.environ.get("VPROC_PORT", "8765")),
        sim_floor=float(os.environ.get("VPROC_SIM_FLOOR", "0.25")),
        hhem_threshold=float(os.environ.get("VPROC_HHEM_THRESHOLD", "0.5")),
        hf_token=os.environ.get("HF_TOKEN"),
    )
