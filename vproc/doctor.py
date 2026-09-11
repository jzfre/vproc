"""Offline installation checks; never import model runtimes or open meeting data."""

from importlib.machinery import PathFinder
from importlib.util import find_spec
import os
from pathlib import Path
import platform
import shutil
import sys

from vproc.config import Config, DEFAULT_DIARIZE_MODEL


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


def run_doctor(cfg: Config) -> int:
    failures = 0

    def report(ok: bool, name: str, success: str, remedy: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {success if ok else remedy}")

    print("vproc doctor: offline local preflight")
    report(sys.version_info >= (3, 12), "Python", platform.python_version(), "Install Python 3.12 or newer.")
    for binary in ("ffmpeg", "ffprobe"):
        report(bool(shutil.which(binary)), binary, "available on PATH",
               "Install FFmpeg (including ffprobe) and add its bin directory to PATH.")

    if cfg.transcribe.base_url:
        print("INFO ASR: configured remote endpoint; not probed")
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
    for name in ("OCR", "embedding", "grounding"):
        print(f"INFO {name}: configured remote endpoint; not probed")
    print("Model weights, downloads, access, runtime loading and remote connectivity were not tested. "
          "Storage checks inspect permissions only; no files were created.")
    print(f"Local preflight {'failed' if failures else 'passed'} ({failures} failure(s)).")
    return 1 if failures else 0
