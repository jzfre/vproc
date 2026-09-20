"""Installation checks. Offline by default: never contacts servers, imports model runtimes or
opens meeting data. `--online` adds small opt-in probes of the configured backends."""

from difflib import get_close_matches
from importlib.machinery import PathFinder
from importlib.util import find_spec
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from urllib.parse import urlsplit

from vproc.config import KNOWN_ENV, Config, DEFAULT_DIARIZE_MODEL

_UNSET = object()


def _module_available(name: str) -> bool:
    # util.find_spec('parent.child') imports parent. Walk package paths ourselves
    # so checking pyannote.audio cannot execute an optional package's __init__.
    parts = name.split(".")
    try:
        spec = find_spec(parts[0])
        for index in range(1, len(parts)):
            if spec is None or spec.submodule_search_locations is None:
                return False
            spec = PathFinder.find_spec(".".join(parts[:index + 1]), spec.submodule_search_locations)
        return spec is not None
    except (ImportError, ValueError):
        return False


def _storage_writable(value: str) -> bool:
    """Check directory permissions without creating files or listing contents."""
    try:
        directory = Path(value).expanduser().resolve()
        while not directory.exists() and directory != directory.parent:
            directory = directory.parent
        return directory.is_dir() and os.access(directory, os.W_OK | os.X_OK)
    except (OSError, RuntimeError):
        return False


def _display_url(base_url: str) -> str:
    """URL without userinfo: credentials embedded in a base URL must never be printed."""
    url = urlsplit(base_url)
    host = url.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    return f"{url.scheme}://{host}{f':{url.port}' if url.port else ''}{url.path}"


def _default_probes() -> dict:
    """Real probes, built lazily so offline doctor never imports a client library."""
    from openai import Timeout

    from vproc.llm import client

    def quick(base_url):
        return client._client(base_url).with_options(timeout=Timeout(15.0, connect=5.0), max_retries=0)

    def models(base_url):
        return [m.id for m in quick(base_url).models.list().data]

    def embed(cfg):
        return len(client.embed_texts(cfg.embed.base_url, cfg.embed.model, ["vproc doctor"])[0])

    def chat(cfg):
        quick(cfg.grounding.base_url).chat.completions.create(
            model=cfg.grounding.model, max_tokens=16, temperature=0.0,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def hub(cfg):
        from huggingface_hub import HfApi

        api = HfApi()
        if cfg.hf_token:
            api.whoami(token=cfg.hf_token)  # an invalid token fails here, whatever the repo
        api.auth_check(cfg.diarize_model, token=cfg.hf_token or False)

    return {"models": models, "embed": embed, "chat": chat, "hub": hub}


def _failure(exc: Exception) -> str:
    """Exception class and HTTP status only: messages can carry URLs, keys or tokens."""
    status = getattr(exc, "status_code", None)
    return type(exc).__name__ + (f" (HTTP {status})" if isinstance(status, int) else "")


def run_doctor(cfg: Config, *, env_file=_UNSET, online: bool = False, probes: dict | None = None) -> int:
    failures = 0

    def report(ok: bool, name: str, success: str, remedy: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {success if ok else remedy}")

    print(f"vproc doctor: {'offline local preflight plus online backend probes' if online else 'offline local preflight'}")
    if env_file is None:
        print(f"WARN config: no .env in {os.getcwd()}; using the process environment and built-in "
              "defaults (including an index in this directory)")
    elif env_file is not _UNSET:
        print(f"INFO config: loaded {env_file}")
    for name in sorted(key for key in os.environ if key.startswith("VPROC_") and key not in KNOWN_ENV):
        close = get_close_matches(name, KNOWN_ENV, n=1)
        print(f"WARN unknown setting {name}: it is ignored"
              + (f"; did you mean {close[0]}?" if close else ""))
    report(sys.version_info >= (3, 12), "Python", platform.python_version(), "Install Python 3.12 or newer.")
    for binary in ("ffmpeg", "ffprobe"):
        report(bool(shutil.which(binary)), binary, "available on PATH",
               "Install FFmpeg (including ffprobe) and add its bin directory to PATH.")

    def endpoint_line(name: str, endpoint) -> None:
        print(f"INFO {name}: {_display_url(endpoint.base_url)} model '{endpoint.model}'; "
              f"{'see probes below' if online else 'not probed'}")

    if cfg.transcribe.base_url:
        endpoint_line("ASR", cfg.transcribe)
    else:
        supported = platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")
        report(supported, "local ASR platform", "Apple Silicon",
               "MLX Whisper requires Apple Silicon macOS. Set VPROC_TRANSCRIBE_BASE_URL "
               "and VPROC_TRANSCRIBE_MODEL for an OpenAI-compatible ASR server.")
        if supported:
            report(_module_available("mlx_whisper"), "local ASR package", "mlx_whisper available",
                   "Install the local-asr extra (see README).")

    for name in ("torch", "transformers"):
        report(_module_available(name), f"answering {name}", "package available",
               f"Install the answer extra (see README); HHEM requires {name}.")

    if cfg.diarize_model:
        report(_module_available("pyannote.audio"), "diarization package", "pyannote.audio available",
               "Install the diarization extra (see README) or set VPROC_DIARIZE_MODEL= to disable diarization.")
        if cfg.diarize_model in (DEFAULT_DIARIZE_MODEL, "pyannote/speaker-diarization-3.1"):
            report(bool(cfg.hf_token and cfg.hf_token.strip()), "HF_TOKEN", "set; gated model access not verified",
                   "Set HF_TOKEN and accept the configured gated model's access conditions on Hugging Face; "
                   "or set VPROC_DIARIZE_MODEL= to disable diarization.")
        else:
            print("INFO diarization access: custom model; credentials and model access not verified")
    else:
        print("INFO diarization: disabled")

    for key, value in (("VPROC_INDEX_PATH", cfg.index_path), ("VPROC_FRAMES_DIR", cfg.frames_dir)):
        report(_storage_writable(value), key, "directory or nearest existing parent appears writable",
               f"Set {key} to a directory with a writable, searchable parent.")
    if cfg.ocr.base_url:
        endpoint_line("OCR", cfg.ocr)
    else:
        print("INFO OCR: disabled (VPROC_OCR_BASE_URL is empty); on-screen text and nameplate "
              "speaker naming are off")
    endpoint_line("embedding", cfg.embed)
    endpoint_line("grounding", cfg.grounding)
    if not online:
        print("Model weights, downloads, access, runtime loading and remote connectivity were not "
              "tested (run 'vproc doctor --online' before a long ingest). "
              "Storage checks inspect permissions only; no files were created.")
        print(f"Local preflight {'failed' if failures else 'passed'} ({failures} failure(s)).")
        return 1 if failures else 0

    from openai import APITimeoutError

    from vproc.llm import client

    client.configure(cfg)
    probes = probes or _default_probes()

    def key_hint(base_url: str) -> str:
        names = [f"VPROC_{n}_API_KEY[_FILE]" for n, ep in (
            ("OCR", cfg.ocr), ("EMBED", cfg.embed), ("GROUNDING", cfg.grounding),
            ("TRANSCRIBE", cfg.transcribe)) if ep.base_url == base_url]
        return " or ".join(names)

    seen: set[str] = set()
    for endpoint in (cfg.transcribe, cfg.ocr, cfg.embed, cfg.grounding):
        if not endpoint.base_url or endpoint.base_url in seen:
            continue
        seen.add(endpoint.base_url)
        served = [ep.model for ep in (cfg.transcribe, cfg.ocr, cfg.embed, cfg.grounding)
                  if ep.base_url == endpoint.base_url]
        name = f"server {_display_url(endpoint.base_url)}"
        try:
            listed = probes["models"](endpoint.base_url)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 404:
                print(f"INFO {name}: reachable; it has no model listing")
            elif status in (401, 403):
                report(False, name, "", f"key rejected, {_failure(exc)}. Set {key_hint(endpoint.base_url)}.")
            else:
                report(False, name, "", f"not reachable, {_failure(exc)}. Check the URL and that the server runs.")
            continue
        missing = [m for m in served if m not in listed]
        if missing:
            print(f"WARN {name}: reachable, but it does not list {', '.join(repr(m) for m in missing)} "
                  "(fine for an alias; otherwise fix the VPROC_*_MODEL value)")
        else:
            report(True, name, "reachable and lists the configured model(s)", "")

    try:
        dimension = probes["embed"](cfg)
        # Never create an index directory from doctor. The store expands "~", so this must too.
        indexed = Path(cfg.index_path).expanduser().exists()
        if indexed:
            from vproc.store.lancedb_store import Store

            Store(cfg.index_path).validate_embedding_model(cfg.embed.model, dimension)
        report(True, "embedding probe", f"{dimension}-dimensional vectors"
               + (", compatible with the index" if indexed else " (no index yet to compare with)"), "")
    except Exception as exc:
        report(False, "embedding probe", "", f"{_failure(exc)}. Ingest and search need a working "
               "embedding server whose model matches the index (see the README on embedding compatibility).")

    started = time.monotonic()
    try:
        probes["chat"](cfg)
        report(True, "grounding probe", f"replied in {time.monotonic() - started:.1f}s", "")
    except APITimeoutError:
        print("WARN grounding probe: no reply within the probe window; the backend is busy or still "
              "loading a model (not counted as a failure)")
    except Exception as exc:
        report(False, "grounding probe", "", f"{_failure(exc)}. Answers need a working chat model.")

    if cfg.diarize_model and os.path.exists(cfg.diarize_model):  # pyannote's own local test
        print("INFO Hugging Face access: local pipeline; hub access not checked")
    elif cfg.diarize_model:
        try:
            probes["hub"](cfg)
            report(True, "Hugging Face access", f"token accepted and {cfg.diarize_model} is accessible", "")
        except ImportError:
            print("INFO Hugging Face access: huggingface_hub is not installed; not checked")
        except Exception as exc:
            report(False, "Hugging Face access", "",
                   f"{_failure(exc)}. Create a valid HF_TOKEN and accept the conditions of "
                   f"{cfg.diarize_model} on its model page, or set VPROC_DIARIZE_MODEL= to disable "
                   "diarization. (An invalid HF_TOKEN can also block other model downloads.)")

    print("Probes used tiny requests only; model quality and local model loading were not tested.")
    print(f"Preflight {'failed' if failures else 'passed'} ({failures} failure(s)).")
    return 1 if failures else 0
