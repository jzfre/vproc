import os
import pathlib
from dataclasses import dataclass

DEFAULT_HHEM_MODEL = "vectara/hallucination_evaluation_model"


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, ignores blanks/comments/malformed lines, and
    does NOT override variables already set in the environment. Call from app entry points
    (cli) so `vproc` works without manually sourcing .env; load_config() itself stays pure."""
    if not os.path.exists(path):
        return
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()  # allow shell-sourceable `export KEY=...`
        if not key:  # e.g. `=foo`; os.environ.setdefault('') raises OSError
            continue
        val = val.strip()
        if val[:1] in ("'", '"'):  # quoted: take the quoted span, drop anything after the close quote
            end = val.find(val[0], 1)
            val = val[1:end] if end != -1 else val[1:]
        else:  # unquoted: drop an inline ` # comment`
            val = val.split(" #", 1)[0].strip()
        os.environ.setdefault(key, val)


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
    hhem_model: str = DEFAULT_HHEM_MODEL
    frames_dir: str = "./vproc_frames"


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
        hhem_model=os.environ.get("VPROC_HHEM_MODEL", DEFAULT_HHEM_MODEL),
        frames_dir=os.environ.get("VPROC_FRAMES_DIR", "./vproc_frames"),
    )
