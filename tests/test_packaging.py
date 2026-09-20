"""Guards on the committed lock: Linux installs must stay on PyTorch's CPU wheels."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _locked_packages(lock_path=ROOT / "uv.lock"):
    return tomllib.loads(Path(lock_path).read_text())["package"]


def cuda_packages(packages) -> list[str]:
    return sorted(p["name"] for p in packages
                  if p["name"].startswith(("nvidia-", "cuda-")) or p["name"] == "triton")


def test_lock_has_no_cuda_stack():
    # vproc has no CUDA code path. PyPI's Linux torch drags in ~5 GB of CUDA libraries, and
    # PyPI's Linux torchcodec cannot load without them, which breaks diarization on a CPU host.
    assert cuda_packages(_locked_packages()) == []


def test_linux_torch_family_is_locked_to_cpu_builds_and_macos_is_untouched():
    versions: dict[str, set[str]] = {}
    for p in _locked_packages():
        if p["name"] in ("torch", "torchaudio", "torchcodec"):
            versions.setdefault(p["name"], set()).add(p["version"])
    for name, found in versions.items():
        assert any(v.endswith("+cpu") for v in found), f"{name} has no +cpu build locked for Linux"
        assert any("+" not in v for v in found), f"{name} lost its PyPI (macOS) build"
    assert set(versions) == {"torch", "torchaudio", "torchcodec"}
