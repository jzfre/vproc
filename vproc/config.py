import os
import pathlib
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

DEFAULT_HHEM_MODEL = "vectara/hallucination_evaluation_model"
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"


def load_dotenv(path: str = ".env") -> str | None:
    """Minimal .env loader: KEY=VALUE lines, ignores blanks/comments/malformed lines, and
    does NOT override variables already set in the environment. Call from app entry points
    (cli) so `vproc` works without manually sourcing .env; load_config() itself stays pure.
    Returns the resolved file it loaded, or None when there was none (doctor reports this:
    a missing .env silently means built-in defaults and an index in the current directory)."""
    if not os.path.exists(path):
        return None
    # UTF-8 with optional BOM supports Windows editors without locale-dependent decoding.
    for line in pathlib.Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line or "\0" in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()  # allow shell-sourceable `export KEY=...`
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        val = val.strip()
        if val[:1] in ("'", '"'):  # quoted: take the quoted span, drop anything after the close quote
            end = val.find(val[0], 1)
            val = val[1:end] if end != -1 else val[1:]
        else:  # unquoted: drop an inline ` # comment`
            val = val.split(" #", 1)[0].strip()
        os.environ.setdefault(key, val)
    return str(pathlib.Path(path).resolve())


_ENDPOINT_PREFIXES = ("VPROC_OCR", "VPROC_EMBED", "VPROC_GROUNDING", "VPROC_TRANSCRIBE")
# Every setting load_config reads, plus VPROC_ASK_URL (read by scripts/ask.sh and search.sh).
# doctor flags other VPROC_* names as probable typos; a test keeps this in step with the
# literals in this file.
KNOWN_ENV = frozenset(
    {f"{prefix}{suffix}" for prefix in _ENDPOINT_PREFIXES
     for suffix in ("_BASE_URL", "_MODEL", "_API_KEY", "_API_KEY_FILE")}
    | {"VPROC_INDEX_PATH", "VPROC_FRAMES_DIR", "VPROC_HOST", "VPROC_PORT", "VPROC_SIM_FLOOR",
       "VPROC_HHEM_THRESHOLD", "VPROC_HHEM_MODEL", "VPROC_OCR_TIMEOUT", "VPROC_OCR_MAX_TOKENS",
       "VPROC_GROUNDING_MAX_TOKENS", "VPROC_GROUNDING_THINKING", "VPROC_DIARIZE_MODEL",
       "VPROC_SPEAKER_NAMING", "VPROC_EMBED_QUERY_PREFIX", "VPROC_ASK_URL"})


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    # Optional bearer key for an authenticated server (e.g. vLLM --api-key). Kept out of
    # repr so a logged or printed Config can never leak it.
    api_key: str | None = field(default=None, repr=False)


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
    ocr_timeout: float = 60.0  # per-frame OCR cap; slow vision backend -> empty screen text, not a stall
    ocr_max_tokens: int = 1024  # bound OCR output so a reasoning vision model can't run away
    grounding_max_tokens: int = 8192  # bound grounding thinking+JSON; a runaway truncates to an abstention
    # Thinking ON = ~60-90s/question on a reasoning model; OFF = seconds. ON won a small A/B on
    # a 9B model (OFF flipped an answerable question to abstention); the trade-off is
    # model-specific, and the HHEM gate and citation rules are identical in both modes.
    grounding_thinking: bool = True
    # Speaker diarization pipeline (HF-gated; uses hf_token). Empty string = disabled.
    diarize_model: str = DEFAULT_DIARIZE_MODEL
    # Nameplate speaker naming (needs diarization + the vision endpoint). Off = keep SPEAKER_xx.
    speaker_naming: bool = True
    # Prepended to search/ask queries before embedding; stored passages are untouched, so
    # changing it needs no re-ingest. Instruction-tuned embedders document one (Qwen3-Embedding:
    # "Instruct: <task>\nQuery: "). It shifts similarity values: re-check sim_floor after changing it.
    embed_query_prefix: str = ""

    def __post_init__(self) -> None:
        for name, minimum, maximum in (("sim_floor", -1, 1), ("hhem_threshold", 0, 1)):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not minimum <= value <= maximum):
                raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
        if (not isinstance(self.ocr_timeout, (int, float)) or isinstance(self.ocr_timeout, bool)
                or not 0 < self.ocr_timeout < float("inf")):
            raise ValueError("ocr_timeout must be positive and finite")
        for name in ("ocr_max_tokens", "grounding_max_tokens", "port"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        for name in ("index_path", "frames_dir", "host", "hhem_model"):
            _required_string(name, getattr(self, name))
        _required_string("diarize_model", self.diarize_model, allow_empty=True)
        if not isinstance(self.embed_query_prefix, str) or "\0" in self.embed_query_prefix:
            raise ValueError("embed_query_prefix must be a string without NUL characters")
        for name in ("ocr", "embed", "grounding", "transcribe"):
            endpoint = getattr(self, name)
            if not isinstance(endpoint, Endpoint):
                raise ValueError(f"{name} must be an Endpoint")
            _required_string(f"{name}.model", endpoint.model)
            # Empty = feature off: local in-process ASR, or no OCR / nameplate naming.
            _required_string(f"{name}.base_url", endpoint.base_url,
                             allow_empty=name in ("transcribe", "ocr"))
            key = endpoint.api_key
            if key is not None and (not isinstance(key, str) or not key
                                    or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in key)):
                raise ValueError(f"{name}.api_key must be a nonblank string without whitespace")
            if key is not None and self.api_key_for(endpoint.base_url) != key:
                raise ValueError("Endpoints sharing a base URL are configured with different API keys")

    def api_key_for(self, base_url: str) -> str | None:
        """A key belongs to a server, not a function: endpoints sharing a base URL (one vLLM
        serving OCR and grounding) share its key, so setting it once is enough."""
        server = base_url.rstrip("/")
        return next((ep.api_key for ep in (self.ocr, self.embed, self.grounding, self.transcribe)
                     if ep.base_url.rstrip("/") == server and ep.api_key is not None), None)


def _required_string(name: str, value: str, *, allow_empty: bool = False) -> None:
    if isinstance(value, str) and "\0" not in value and (value.strip() or (allow_empty and value == "")):
        return
    raise ValueError(f"{name} must be a nonblank string without NUL characters")


def _api_key(prefix: str) -> str | None:
    """Inline key, or a file so the secret can live outside .env (e.g. a 0600 file shared
    with other clients). Errors name the setting, never the value."""
    key, file_key = f"{prefix}_API_KEY", f"{prefix}_API_KEY_FILE"
    value, path = os.environ.get(key, "").strip(), os.environ.get(file_key, "").strip()
    if value and path:
        raise ValueError(f"Set only one of {key} and {file_key}")
    if path:
        try:
            value = pathlib.Path(path).expanduser().read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError):
            raise ValueError(f"{file_key} must name a readable UTF-8 text file") from None
        if not value:
            raise ValueError(f"{file_key} names an empty file")
    return value or None


def _ep(prefix: str, default_url: str, default_model: str) -> Endpoint:
    key = f"{prefix}_BASE_URL"
    base_url = os.environ.get(key, default_url)
    if not (prefix in ("VPROC_TRANSCRIBE", "VPROC_OCR") and base_url == ""):
        # Validate deployment URLs here; direct Config callers may inject local/fake clients.
        try:
            url = urlsplit(base_url)
            valid = (url.scheme in ("http", "https") and bool(url.hostname)
                     and (url.port is None or 1 <= url.port <= 65535)
                     and not url.query and not url.fragment
                     and not any(char.isspace() or ord(char) < 32 or ord(char) == 127
                                 for char in base_url))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"{key} must be an http/https base URL without a query or fragment")
    return Endpoint(base_url, os.environ.get(f"{prefix}_MODEL", default_model), _api_key(prefix))


def _bool_env(key: str, default: str = "on") -> bool:
    value = os.environ.get(key, default).strip().lower()
    if value in ("1", "true", "on", "yes"):
        return True
    if value in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"{key} must be on/off, true/false, yes/no, or 1/0")


def _number_env(key: str, default: str, number_type: type[int] | type[float]) -> int | float:
    try:
        return number_type(os.environ.get(key, default))
    except ValueError:
        expected = "an integer" if number_type is int else "a number"
        raise ValueError(f"{key} must be {expected}") from None


def load_config() -> Config:
    return Config(
        ocr=_ep("VPROC_OCR", "http://localhost:8000/v1", "QuantTrio/Qwen3.5-9B-AWQ"),
        embed=_ep("VPROC_EMBED", "http://localhost:1234/v1", "Qwen3-Embedding-0.6B"),
        grounding=_ep("VPROC_GROUNDING", "http://localhost:1234/v1", "Qwen3-32B-AWQ"),
        transcribe=_ep("VPROC_TRANSCRIBE", "", DEFAULT_TRANSCRIBE_MODEL),
        index_path=os.environ.get("VPROC_INDEX_PATH", "./vproc.lance"),
        host=os.environ.get("VPROC_HOST", "0.0.0.0"),
        port=_number_env("VPROC_PORT", "8765", int),
        sim_floor=_number_env("VPROC_SIM_FLOOR", "0.25", float),
        hhem_threshold=_number_env("VPROC_HHEM_THRESHOLD", "0.5", float),
        hf_token=os.environ.get("HF_TOKEN"),
        hhem_model=os.environ.get("VPROC_HHEM_MODEL", DEFAULT_HHEM_MODEL),
        frames_dir=os.environ.get("VPROC_FRAMES_DIR", "./vproc_frames"),
        ocr_timeout=_number_env("VPROC_OCR_TIMEOUT", "60", float),
        ocr_max_tokens=_number_env("VPROC_OCR_MAX_TOKENS", "1024", int),
        grounding_max_tokens=_number_env("VPROC_GROUNDING_MAX_TOKENS", "8192", int),
        grounding_thinking=_bool_env("VPROC_GROUNDING_THINKING"),
        diarize_model=os.environ.get("VPROC_DIARIZE_MODEL", DEFAULT_DIARIZE_MODEL),
        speaker_naming=_bool_env("VPROC_SPEAKER_NAMING"),
        # .env values are single-line: a literal "\n" stands for the newline such prefixes need.
        embed_query_prefix=os.environ.get("VPROC_EMBED_QUERY_PREFIX", "").replace("\\n", "\n"),
    )
